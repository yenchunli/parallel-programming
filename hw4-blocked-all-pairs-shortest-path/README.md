# Parallel Programming HW4-1

106062209 Yen-Chun Li

## Implementation

### a. How do you divide your data?

First, I add the additional data to make the length of data is divided by 64. For example, if the total metrics is 82, we will add the additional data to make its length become 64*2 = **128**.

### b. What’s your configuration? And why? (e.g. blocking factor, #blocks, #threads)

I choose 64 as my blocking factors.

The configuration of my code is in Table.1:

| Kernel function | Grid Size             | Block Size |
| --------------- | --------------------- | ---------- |
| Phase1          |     1                 | (32, 32)   |
| Phase2_1        | (1, Round-1)          | (32, 32)   |
| Phase2_2        | (Round-1, 1)          | (32, 32)   |
| Phase3          | (Round-1 Round - 1)   | (32, 32)   |

> Table 1. The configuration in phase1,2,3

- Phase1 only have 1 block each round, so we need 1 Grid.
- Phase2 has two types of data, row direction and column direction, so I split it into two function in order to get better illusion
- Phase3 is the remaining part of reactangle, so we need (Round-1,Round-1) Grids.

There are two constraints on how you choose the block size:

1. Maximum number of threads per block

For GTX 1080, the maximum number of threads per block is 1024 (32 * 32). If we set blockSize=64, it needs 64 * 64 = **4096** threads to execute whole programs in one block. Unfortunately, it exceed the maximum number of threads per block. My solution is to divide a 64 * 64 block into four 32 * 32 block. In each iteration, it select four elements in the block to execute. By doing so, we can utilize Instruction Level Parallelism (ILP) to process four data in one thread.

2. Share memory per block

For GTX 1080, it provides 49KB share memory per block. It represents that we can store 49000/sizeof(int) = **12000** data. In Phase 3, we need 64 * 64 * 2 = **8192** data in each round. 8192 < 12000, it doesn't exceed the maximum share memory per block.

```c
//The width of the block is 64*64, and we divide into four 32*32 block.
// * means the data that might be executed in first iteration.
// Their index is (i,j), (i+32,j), (i,j+32), (i+32,j+32)
 ----- -----
| *   |*    |
|     |     |
 ----- -----
|*    |*    |
|     |     |
 ----- -----
```

![maximum number of threads per block](https://i.imgur.com/GJLRrJs.png)
![Total Share memory per block](https://i.imgur.com/t0Fv4OA.png)
![](https://i.imgur.com/3p2D2OB.png)
Image source: All-Pairs-Shortest-Paths for Large Graphs on the GPU Gary J Katz 1,2, Joe Kider 1 1 University of Pennsylvania 2 Lockheed Martin IS&GS. https://slideplayer.com/slide/5313565/

### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cuda.h>
#include <chrono>
#include <unistd.h>
#include <iostream>

const int INF = ((1 << 30) - 1); 
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
template <int,int> __global__ void phase1(int* Dist_gpu, int Round, int n, int pitch_int);
template <int,int> __global__ void phase2_1(int* Dist_gpu, int Round, int n, int pitch_int);
template <int,int> __global__ void phase2_2(int* Dist_gpu, int Round, int n, int pitch_int);
template <int,int> __global__ void phase3(int* Dist_gpu, int Round, int n, int pitch_int);

int n, m;   // Number of vertices, edges
int* Dist;
int* Dist_gpu;
size_t pitch;

int N;

int main(int argc, char* argv[])
{   
    
    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    cudaFreeHost(Dist);
    cudaFree(Dist_gpu);
    return 0;
}

void input(char* infile) { 
    FILE* file = fopen(infile, "rb"); 
    fread(&n, sizeof(int), 1, file); 
    fread(&m, sizeof(int), 1, file);

    N = n;
    n = (!n%64)? n : n + 64 - n%64;

    cudaMallocHost( &Dist, sizeof(int)*(n*n));

    for (int i = 0; i < n; ++i) {
        int IN = i * n;
        #pragma GCC ivdep
        for (int j = 0; j < i; ++j) {
            Dist[IN + j] = INF;
        }
        #pragma GCC ivdep
        for (int j = i + 1; j < n; ++j) {
            Dist[IN + j] = INF;
        }
    }

    int pair[3]; 
    for (int i = 0; i < m; ++i) { 
        fread(pair, sizeof(int), 3, file); 
        Dist[pair[0] * n + pair[1]] = pair[2]; 
    } 
    fclose(file); 
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < N; ++i) {
        fwrite(&Dist[i * n], sizeof(int), N, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) {
    return (a + b -1)/b;
}

void block_FW(int B)
{

    int round = ceil(n, B);
    cudaMallocPitch((void**)&Dist_gpu, &pitch,n*sizeof(int), n+64);
    int pitch_int = pitch / sizeof(int);
    cudaMemcpy2D(Dist_gpu, pitch, Dist, n*sizeof(int), n*sizeof(int), n, cudaMemcpyHostToDevice);
    
    for (int r = 0; r < round; ++r) {
        
        switch(B){
            case 32:
            break;
            case 64:
                phase1  <64,4><<< 1                     , dim3(32,32),   64*64*sizeof(int) >>>(Dist_gpu, r, n, pitch_int);
                phase2_1<64,4><<< dim3(1, round-1)      , dim3(32,32), 2*64*64*sizeof(int) >>>(Dist_gpu, r, n, pitch_int);
                phase2_2<64,4><<< dim3(round-1, 1)      , dim3(32,32), 2*64*64*sizeof(int) >>>(Dist_gpu, r, n, pitch_int);
                phase3  <64,4><<< dim3(round-1, round-1), dim3(32,32), 2*64*64*sizeof(int) >>>(Dist_gpu, r, n, pitch_int);
            break;
        }
        
        
        
    }
    cudaMemcpy2D(Dist, n*sizeof(int), Dist_gpu, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
}

template <int B, int P>
__global__ 
void phase1(int* Dist_gpu, int Round, int n, int pitch_int) {

    extern __shared__ int shared_mem[]; 

    int sdx = (threadIdx.y * 64) + threadIdx.x;

    shared_mem[sdx]      = Dist_gpu[(Round * 64 + threadIdx.y)     *pitch_int + Round * 64 + threadIdx.x];
    shared_mem[sdx+32]   = Dist_gpu[(Round * 64 + threadIdx.y)     *pitch_int + Round * 64 + threadIdx.x + 32];
    shared_mem[sdx+2048] = Dist_gpu[(Round * 64 + threadIdx.y + 32)*pitch_int + Round * 64 + threadIdx.x];
    shared_mem[sdx+2080] = Dist_gpu[(Round * 64 + threadIdx.y + 32)*pitch_int + Round * 64 + threadIdx.x + 32];

    //__syncthreads();

    for(int k=0; k < 64; ++k){
        __syncthreads();
        shared_mem[sdx]      = min(shared_mem[sdx]     , shared_mem[threadIdx.y * 64 + k]    + shared_mem[k*64+threadIdx.x]);
        shared_mem[sdx+32]   = min(shared_mem[sdx+32]  , shared_mem[threadIdx.y * 64 + k]    + shared_mem[k*64+threadIdx.x + 32]);
        shared_mem[sdx+2048] = min(shared_mem[sdx+2048], shared_mem[(threadIdx.y+32)*64 + k] + shared_mem[k*64+threadIdx.x]);
        shared_mem[sdx+2080] = min(shared_mem[sdx+2080], shared_mem[(threadIdx.y+32)*64 + k] + shared_mem[k*64+threadIdx.x + 32]);
    }

    Dist_gpu[(Round * 64  + threadIdx.y)     *pitch_int + Round * 64 + threadIdx.x]       = shared_mem[sdx];
    Dist_gpu[(Round * 64  + threadIdx.y)     *pitch_int + Round * 64 + threadIdx.x + 32]  = shared_mem[sdx+32];
    Dist_gpu[(Round * 64  + threadIdx.y + 32)*pitch_int + Round * 64 + threadIdx.x]       = shared_mem[sdx+2048];
    Dist_gpu[(Round * 64  + threadIdx.y + 32)*pitch_int + Round * 64 + threadIdx.x + 32]  = shared_mem[sdx+2080];
}

template <int B, int P>
__global__ void phase2_1(int* Dist_gpu, int Round, int n, int pitch_int) {
    
    extern __shared__ int shared_mem[]; 

    int b_i = blockIdx.y + (blockIdx.y >= Round);
    int b_j = Round;

    int i = b_i * 64 + threadIdx.y;
    int j = b_j * 64 + threadIdx.x;

    int sdx = threadIdx.y * 64 + threadIdx.x;

    shared_mem[sdx]                    = Dist_gpu[i                             * pitch_int + j      ]; // IK
    shared_mem[sdx + 32]               = Dist_gpu[i                             * pitch_int + j + 32 ]; // IK
    shared_mem[sdx + 4096]             = Dist_gpu[(Round*64 + threadIdx.y)      * pitch_int + j      ]; // KJ
    shared_mem[sdx + 4096 + 32]        = Dist_gpu[(Round*64 + threadIdx.y)      * pitch_int + j + 32 ]; // KJ

    shared_mem[sdx + 2048]             = Dist_gpu[(i + 32)                      * pitch_int + j      ];
    shared_mem[sdx + 2048 + 32]        = Dist_gpu[(i + 32)                      * pitch_int + j + 32 ];
    shared_mem[sdx + 2048 + 4096]      = Dist_gpu[(Round*64 + threadIdx.y + 32) * pitch_int + j      ];
    shared_mem[sdx + 2048 + 4096 + 32] = Dist_gpu[(Round*64 + threadIdx.y + 32) * pitch_int + j + 32 ];

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();
        
        shared_mem[sdx]      = min(shared_mem[sdx],      shared_mem[threadIdx.y*B+k] + shared_mem[k*B+threadIdx.x + 4096]);
        shared_mem[sdx + 32] = min(shared_mem[sdx + 32], shared_mem[threadIdx.y*B+k] + shared_mem[k*B+threadIdx.x + 4096 + 32]);

        shared_mem[sdx + 2048] = min(shared_mem[sdx+2048], shared_mem[(threadIdx.y + 32)*64+k] + shared_mem[k*64+threadIdx.x + 4096]);
        shared_mem[sdx +2048 + 32] = min(shared_mem[sdx + 2048 + 32], shared_mem[(threadIdx.y + 32)*64+k] + shared_mem[k*64+threadIdx.x + 4096 + 32]);
    }

    Dist_gpu[i       *pitch_int + j     ] = shared_mem[sdx            ];  
    Dist_gpu[i       *pitch_int + j + 32] = shared_mem[sdx + 32       ];
    Dist_gpu[(i + 32)*pitch_int + j     ] = shared_mem[sdx + 2048     ];  
    Dist_gpu[(i + 32)*pitch_int + j + 32] = shared_mem[sdx + 2048 + 32]; 

}

template <int B, int P>
__global__ void phase2_2(int* Dist_gpu, int Round, int n, int pitch_int) {
    extern __shared__ int shared_mem[];

    int i = (Round << 6) + threadIdx.y;
    int j = ((blockIdx.x + (blockIdx.x >= Round)) << 6) + threadIdx.x;
    
    int sdx = threadIdx.y * 64 + threadIdx.x;

    shared_mem[sdx]                    = Dist_gpu[i*pitch_int + j];
    shared_mem[sdx + 32]               = Dist_gpu[i*pitch_int + j + 32];
    shared_mem[sdx + 4096]             = Dist_gpu[i*pitch_int + Round * 64 + threadIdx.x];
    shared_mem[sdx + 4096 + 32]        = Dist_gpu[i*pitch_int + Round * 64 + threadIdx.x + 32];

    shared_mem[sdx + 2048]             = Dist_gpu[(i + 32)*pitch_int + j];
    shared_mem[sdx + 2048 + 32]        = Dist_gpu[(i + 32)*pitch_int + j + 32];
    shared_mem[sdx + 2048 + 4096]      = Dist_gpu[(i + 32)*pitch_int + Round * 64 + threadIdx.x];
    shared_mem[sdx + 2048 + 4096 + 32] = Dist_gpu[(i + 32)*pitch_int + Round * 64 + threadIdx.x + 32];

    

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();

        
        shared_mem[sdx] = min(shared_mem[sdx], shared_mem[threadIdx.y*64+k+4096] + shared_mem[k*64+threadIdx.x]);
        shared_mem[sdx + 32] = min(shared_mem[sdx + 32], shared_mem[threadIdx.y*64+k+4096] + shared_mem[k*64+threadIdx.x + 32]);

        
        shared_mem[sdx+2048] = min(shared_mem[sdx+2048], shared_mem[(threadIdx.y+32)*64+k+4096] + shared_mem[k*64+threadIdx.x]);
        shared_mem[sdx+2048 + 32] = min(shared_mem[sdx+2048 + 32], shared_mem[(threadIdx.y+32)*64+k+4096] + shared_mem[k*64+threadIdx.x + 32]);
        
    }

    Dist_gpu[i*pitch_int + j]      = shared_mem[sdx];  
    Dist_gpu[i*pitch_int + j + 32] = shared_mem[sdx + 32];

    Dist_gpu[(i + 32)*pitch_int + j]      = shared_mem[sdx+2048];  
    Dist_gpu[(i + 32)*pitch_int + j + 32] = shared_mem[sdx+2048 + 32];
}

template <int B, int P>
__global__ void phase3(int* Dist_gpu, int Round, int n, int pitch_int) {

    int i = ((blockIdx.y + (blockIdx.y>=Round)) << 6) + threadIdx.y;
    int j = ((blockIdx.x + (blockIdx.x>=Round)) << 6) + threadIdx.x;

    extern __shared__ int shared_mem[];
    
    int d0 = Dist_gpu[i*pitch_int + j];
    int d1 = Dist_gpu[i*pitch_int + j + 32];
    int d2 = Dist_gpu[(i+32)*pitch_int + j];
    int d3 = Dist_gpu[(i+32)*pitch_int + j + 32];
    
    int sdx = threadIdx.y * 64 + threadIdx.x;

    shared_mem[ sdx ]       = Dist_gpu[i*pitch_int + Round * 64 + threadIdx.x];
    shared_mem[ sdx + 32]   = Dist_gpu[i*pitch_int + Round * 64 + threadIdx.x + 32];
    shared_mem[ sdx + 4096] = Dist_gpu[(Round * 64 + threadIdx.y)*pitch_int + j];
    shared_mem[ sdx + 4128] = Dist_gpu[(Round * 64 + threadIdx.y)*pitch_int + j + 32];

    sdx += 2048;

    shared_mem[ sdx ]       = Dist_gpu[(i + 32)*pitch_int + Round * 64 + threadIdx.x];
    shared_mem[ sdx + 32]   = Dist_gpu[(i + 32)*pitch_int + Round * 64 + threadIdx.x + 32];
    shared_mem[ sdx + 4096] = Dist_gpu[(Round * 64 + (threadIdx.y + 32))*pitch_int + j];
    shared_mem[ sdx + 4128] = Dist_gpu[(Round * 64 + (threadIdx.y + 32))*pitch_int + j + 32];

    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        int idx = threadIdx.y * 64 + k;
        int v0 = shared_mem[idx]        + shared_mem[k*64 + threadIdx.x + 4096];
        int v1 = shared_mem[idx]        + shared_mem[k*64 + threadIdx.x + 4128];
        int v2 = shared_mem[idx + 2048] + shared_mem[k*64 + threadIdx.x + 4096];
        int v3 = shared_mem[idx + 2048] + shared_mem[k*64 + threadIdx.x + 4128];
        d0 = min(d0, v0);
        d1 = min(d1, v1);
        d2 = min(d2, v2);
        d3 = min(d3, v3);
    }

    Dist_gpu[i*pitch_int + j]           = d0;
    Dist_gpu[i*pitch_int + j + 32]      = d1;
    Dist_gpu[(i+32)*pitch_int + j]      = d2;
    Dist_gpu[(i+32)*pitch_int + j + 32] = d3;

}
```

## Profiling Results

1. Occupancy
2. SM Efficiency
3. Shared Memory Load/Store Throughput
4. Global Load/Store Throughput

![Profiling Results](https://i.imgur.com/p60T9b8.png)


## Experiment & Analysis

### System Spec

I use hades.cs.nthu.edu.tw to test my program

### Time Distribution

I use the following code to measure time.

```c
#include <iostream>
#include <chrono>
std::chrono::steady_clock::time_point t1;
std::chrono::steady_clock::time_point t2;
t1 = std::chrono::steady_clock::now();
input(argv[1]);
t2 = std::chrono::steady_clock::now();
std::cout << "[Input] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";
```

The command which can show the time of multiple kernel function.

```c
srun -p prof -N1 -n1 --gres=gpu:1 nvprof ./hw4-1 /home/pp20/share/hw4-1/cases/c21.1 /dev/shm/c21.out
```

- computing

- memory copy (H2D, D2H)

- I/O of your program w.r.t. input size.

|Testcase|n|Total(s)|Input(s)|Output(s)|Compute(s)|H2D(s)|D2H(s)|
|-|-|-|-|-|-|-|-|
| c21      |  5000 | 1.144	|0.739|0.060|0.140	|0.017	|0.015|
| p11k1    | 11000 | 2.224	|0.462|0.165|1.236	|0.081	|0.073|
| p16k1    | 16000 | 5.684	|0.882|0.352|3.857	|0.172	|0.157|
| p21k1    | 20959 | 11.078	|0.998|0.614|8.581	|0.293	|0.267|
| p26k1    | 25899 | 19.786	|1.342|0.923|16.245	|0.447	|0.362|

![](https://i.imgur.com/ThMteR5.png)


![Time distribution, N=5000, c21](https://i.imgur.com/JImVDS7.png)

![Time distribution, N=11000, p11k1](https://i.imgur.com/IplIHyo.png)

![Time distribution, N=16000, p16k1](https://i.imgur.com/E4fSVC1.png)

![Time distribution, N=20959, p21k1](https://i.imgur.com/2EcnmAn.png)

![Time distribution, N=25899, p26k1](https://i.imgur.com/t4sBuE4.png)

### Blocking Factor

I choose c21.1 to be the testbed because of the speed. And for convenience, I only select phase3 because its took most of the time.

|Blocking factor|gld throughput (GB/s)|gst throughput (GB/s)|shared load throughtput (GB/s)|shared store throughput (GB/s)|	Integer Instructions | GOPS |
|-|-|-|-|-|-|-|
| 8|      |      |      |      |          |         |
|16|295.91|98.636|2465.9|394.54| 312289409|1214.724 |
|32|187.11|62.369|3056.1|249.48| 465397183|1187.842|
|64|199.30|66.433|3188.8|132.87|2853347328|2031.859 |

![Global Memory Access](https://i.imgur.com/Cn1tn0C.png)
![Share Memory Access](https://i.imgur.com/omIX1OE.png)
![c21.1 total time](https://i.imgur.com/2Ce3yRC.png)





- Profiling blockSize=64
![Profiling blockSize=64](https://i.imgur.com/GDCWtKd.png)
![](https://i.imgur.com/BSbVgi8.png)

- Profiling blockSize=32
![Profiling blockSize=32](https://i.imgur.com/wU4F7YN.png)
![](https://i.imgur.com/0H1uS6v.png)


- Profiling blockSize=16
![Profiling blockSize=16](https://i.imgur.com/G0GCZPJ.png)
![](https://i.imgur.com/7htNd1b.png)


- Profiling blockSize=8
Always be canceled by srun QQ

I also measure the speed on p11k1 because there might be some error when the # nodes is small.

|Blocking factor| Total Time |
|-|-|
|8  |8.453168|
|16 |4.325353|
|32 |3.707174|
|64 |2.246268|

![p11k1, total time](https://i.imgur.com/b4KT5PS.png)





## Optimization

|Optimization           |Total Time |Speedup    |Step Speedup|
|-|-|-|-|
|base	                |427.82724	|1.000	    |1|
|coaleaced	            |45.165043	|9.473	    |9.472530337|
|padding	            |35.049087	|12.207	    |1.288622525|
|share memory	        |12.673433	|33.758	    |2.765555868|
|large blocking factor	|3.922478	|109.071	|3.230976184|
|unroll	                |3.677828	|116.326	|1.06652024|


![Total time](https://i.imgur.com/2rvKA5M.png)
![Speedup](https://i.imgur.com/zsThCIn.png)




## Experience & Conclusion

這次作業學到了如何優化CUDA，會記得先處理memory的sequential access來增加速度，並且使用shared memory的技巧來加速。

作業截止前一天hades好擠，都擠不進去，希望下次要早一點想到方法。

## Scripts

### test bank conflict

```
srun -p prof -N1 -n1 --gres=gpu:1 nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./hw4-1 ./cases/c09.1 test.out
```

### test all

```
srun -p prof -N1 -n1 --gres=gpu:1 nvprof -m all ./hw4-1 ./cases/c09.1 test.out
```

### test throughput

```
srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics gld_throughput ./hw4-1 ./cases/c09.1 test.out
```

### test occupancy

```
srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy ./hw4-1 ./cases/c09.1 test.out
```

### test time profiling

```
srun -p prof -N1 -n1 --gres=gpu:1 nvprof -m sm_efficiency,achieved_occupancy,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput ./hw4-1 /home/pp20/share/hw4-1/cases/c21.1 /dev/shm/c21.1.out
```

### test integer instruction ...

```
srun -p prof -N1 -n1 --gres=gpu:1 nvprof -m inst_integer,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput ./hw4-1 /home/pp20/share/hw4-1/cases/c21.1 /dev/shm/c21.1.out
```
