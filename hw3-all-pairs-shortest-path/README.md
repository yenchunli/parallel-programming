# Parallel Programming HW3

106062209 Yen-Chun Li

## Implementation

### Which algorithm do you choose?

我使用Block Floyd Warshall來實作。

### Describe your implementation.

首先這次我使用了clang編譯，發現clang可以幫我做vectorize。下面是這次使用的makefile:

我加入了一些參數，發現加入前後是對於效能有影響的。

* msse4 msse3 msse2: 指令集的開啟編譯
* -Rpass-missed=loop-vectorize: 告訴我哪些for loop是vectorized 失敗的。
* -Rpass-analysis=loop-vectorize: 分析loop vectorized的為什麼成功or失敗。
* -Rpass=loop-vectorize: 強制會跑loop vectorize。

![clang vectorization](https://i.imgur.com/Ny9vpz4.png)

```shell
CC = gcc
CXX = clang++
CXXFLAGS = -O3 -fopenmp -msse4 -msse2 -msse3 -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize
CFLAGS = -O3 -lm -pthread -fopenmp
TARGETS = hw3

ifeq ($(TIME),1)
	CXXFLAGS += -DTIME
endif

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
```

同時我也有嘗試手動vectorized，但是發現效能沒有比auto vectorized好，大概是跑hw3-judge時21秒對17秒的差距，下面是手動的vectorized的code。

```c
int block_internal_start_y = b_j * B;
int block_internal_end_y = (b_j + 1) * B;
block_internal_end_y =  (block_internal_end_y > n)? n : block_internal_end_y;


if (block_internal_end_y > n) {
    //block_internal_end_y = n;
    block_internal_start_y = n-B;
}


for (int k = RB; k < RBB && k < n; ++k) {
    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2


    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        /*
        for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
           int r = Dist[i][k] + Dist[k][j];
           int l = Dist[i][j];
           Dist[i][j] = r * (r < l) + l * (r >= l);
        }
        */

        int j = 0;

        __m128i IK =_mm_set1_epi32(Dist[i][k]);
        for (; j < B; j += 4) {
            int jdx = j+block_internal_start_y;
            __m128i left = _mm_lddqu_si128((__m128i*)&(Dist[i][jdx]));
            __m128i right = _mm_lddqu_si128((__m128i*)&(Dist[k][jdx]));
            right = _mm_add_epi32(IK, right);
            __m128i compare1 = _mm_cmplt_epi32 (right, left);
            __m128i compare2 = _mm_andnot_si128(compare1, left);
            compare1 = _mm_and_si128(right, compare1);
            left = _mm_add_epi32(compare1, compare2);
            //edge = _mm_or_si128(_mm_and_si128(compare, val), _mm_andnot_si128(compare, edge));
            _mm_storeu_si128((__m128i*)&(Dist[i][jdx]), left);
        } 

    }
}
}
```

這次交的code是用clang來做auto-vectorized，解釋的部分打在程式碼中的註解裡。

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

#ifdef TIME
    #include <chrono>
    double time1;
    double comm_time = 0;
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    double omp_timer[12]; 
#endif

int num_of_threads;
const int INF = 1073741823;
const int V = 6010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_of_threads = CPU_COUNT(&cpu_set);
    omp_set_num_threads(num_of_threads);
    input(argv[1]);
    /********************************************
     * 我使用blockSize 64，因為之後的實驗發現64最快。
     *******************************************/
    int B = 64;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {

#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    /********************************************
     * 這裡使用分割的方式，用i的index讓j的for loop變成
     * 前後兩塊，讓compiler能vectorized。
     *******************************************/
     
    for (int i = 0; i < n; ++i) {
        #pragma clang loop vectorize(enable) interleave(enable)
        for (int j = 0; j < i; ++j) {
            Dist[i][j] = INF;
        }
        Dist[i][i] = 0;
        #pragma clang loop vectorize(enable) interleave(enable)
        for (int j = i + 1; j < n; ++j) {
            Dist[i][j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Input] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";
#endif

}

void output(char* outFileName) {

#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Output] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";
#endif

}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {

#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

    int round = ceil(n, B);
    
    for (int r = 0; r < round; ++r) {
        /********************************************
         * Phase1: 從左上角沿著對角線慢慢做到右下角，長度為1的block。
         *******************************************/
        cal(B, r, r, r, 1, 1);

        /********************************************
         * Phase2: 跟phase1所操作的block位於同一row或column。
         *******************************************/
        cal(B, r,       r,      0,                  r,              1);
        cal(B, r,       r,  r + 1,      round - r - 1,              1);
        cal(B, r,       0,      r,                  1,              r);
        cal(B, r,   r + 1,      r,                  1,  round - r - 1);

        /********************************************
         * Phase3: 剩下的區域。
         *******************************************/
        cal(B, r,       0,      0,                  r,              r);
        cal(B, r,       0,  r + 1,      round - r - 1,              r);
        cal(B, r,   r + 1,      0,                  r,  round - r - 1);
        cal(B, r,   r + 1,  r + 1,      round - r - 1,  round - r - 1);
    }

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Compute] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";

    for(int i=0; i<num_of_threads; ++i)
        printf("[thread_id=%d]%f\n", i, omp_timer[i]);
#endif

}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    
    /********************************************
     * 先把一些重複計算的東西提到最外面，避免在裡面的重複計算。
     *******************************************/
    int RB = Round * B;
    int RBB = (Round+1) * B;
    if(RBB > n) RBB = n;
    
    /********************************************
     * 使用 omp 來加速 for loop。
     *******************************************/
    #pragma omp parallel for schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {

#ifdef TIME
    double omp_t1 = omp_get_wtime();
#endif
        /********************************************
         * 將可以在外層先計算好的算式先行計算完成。
         *******************************************/
        int block_internal_start_x = b_i * B;
        int block_internal_end_x = (b_i + 1) * B;
        block_internal_end_x =  (block_internal_end_x > n)? n : block_internal_end_x;
        
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;
            block_internal_end_y =  (block_internal_end_y > n)? n : block_internal_end_y;

            for (int k = RB; k < RBB; ++k) {
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    /********************************************
                     * 先提出Dist[i][k]，避免一直存取memory。
                     *******************************************/
                    int IK = Dist[i][k];
                    #pragma clang loop vectorize(enable) interleave(enable)
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                       /********************************************
                         * 運用GPU避免condition的方法來改寫if cocndition。
                         *******************************************/
                       int l = IK + Dist[k][j];
                       int r = Dist[i][j];
                       Dist[i][j] = l * (l < r) + r * (l >= r);
                    }
                }
            }
        }

#ifdef TIME
    omp_timer[omp_get_thread_num()] += omp_get_wtime() - omp_t1;
#endif

    }
}
```
### What is the time complexity of your algorithm, in terms of number of vertices V, number of edges E, number of CPUs P?

O(V^3/P)，因為工作基本上還是照著floyd warhsall( O(V^3) )去做，只是分給了p個CPU平行去運算，所以是O(V^3/p)。

### How did you design & generate your test case?

主要是透過random的方式來產生測資，每兩次會產生一組edge，其大小介於501到1000之間。
```c
int main(int argc, char** argv) {

    srand(time(NULL));
    m = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(i != j) {
                int random = rand();
                if( random % 2 == 0 ) {
                    int _distance = ( rand() % 500) + 501;
                    (edges[m]).src = i;
                    (edges[m]).dst = j;
                    (edges[m]).distance = _distance;
                    m++;
                }
                
            }
        }
    }

    std::random_shuffle(edges, edges + m);
    output(argv[1]);
}
```

## Experiment & Analysis

### System Spec

使用apollo來跑實驗。

### Performance Metrics

測量時間是使用 `chrono` , `omp_get_wtime()` 和 `srun time`。使用方式為:

#### srun time

```shell=
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 6
echo "b128"
srun time ./hw3 ./cases/c20.1 c20.1.out 128
```

#### chrono

```c
#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

/*******************************
 * 想要測量的程式碼。
 * *****************************
 */
#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Compute] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";

    for(int i=0; i<num_of_threads; ++i)
        printf("[thread_id=%d]%f\n", i, omp_timer[i]);
#endif
```

#### omp_get_wtime()
```c
#ifdef TIME
    double omp_t1 = omp_get_wtime();
#endif

/*******************************
 * 想要測量的程式碼。
 * *****************************
 */
 
#ifdef TIME
    omp_timer[omp_get_thread_num()] += omp_get_wtime() - omp_t1;
#endif

#ifdef TIME
    for(int i=0; i<num_of_threads; ++i)
        printf("[thread_id=%d]%f\n", i, omp_timer[i]);
#endif
```

程式執行的結果如下，我測量了以下的時間:

1. Input: 讀取檔案並初始化的時間(us)
2. Compute: 總共運算的時間(us)
3. OMP thread各別執行的時間(s)
4. Output: 寫入檔案的時間(us)
5. 整個程式總共花費的時間(s)

![testcase結果](https://i.imgur.com/H4OYIAX.png)

### Strong scalability & Time profile

我做了兩個實驗，第一個是測試在不同的thread數量時的效能變化。可以發現大致上效能是隨著thread的數量增加而上升的，值得注意的是，寫入與讀取的時間，不太穩定，可能要多做幾次才知道是不是誤差。另外在thread num = 6 時，可以看到它分配任務是不太平均的。

|c	|INPUT(s)   |Compute(s) 	|output	(s)     |total	(s) |thread avg	 (s)    |thread time stdev|Speedup|
|-|-|-|-|-|-|-|-|
|1	|0.629	|48.148	|0.200|48.98|48.133	|0.000	|1.000|
|2	|0.630	|24.408	|0.207|25.25|24.092	|0.164	|1.940|
|4	|0.625	|12.585	|0.214|13.43|12.105	|0.138	|3.647|
|6	|0.619	|13.460	|0.239|14.33|9.285	|1.610	|3.418|
|8	|0.644	|7.146	|0.207|8.010|6.386	|0.190	|6.115|
|10|0.641	|5.702	|0.212|6.580|5.069	|0.106	|7.444|
|12|0.670	|4.918	|0.241|5.850|4.268	|0.097	|8.373|

![Total Time on Different Thread Number](https://i.imgur.com/zxBKrr8.png)
![Speedup on Different Thread Number](https://i.imgur.com/dAr2vB3.png)

第二個是不同的blockSize會不會影響時間。我們可以看到在blockSize=64時的時候速度最快。另外當blocksize變大時，thread的分配狀況會變得較不平均，推測是因為block變大後，每一個任務的執行時間變長，執行時間較容易出現差距所導致。


|blockSize	|INPUT	    |Compute	|output	    |total	|thread avg	   |thread time stdev|Speedup|
|-|-|-|-|-|-|-|-|
|4		|0.649|	40.116	|0.233|	41.020|	39.637	|0.045|	1.000|
|8		|0.641|	17.225	|0.209|	18.090|	16.894	|0.049|	2.268|
|16		|0.665|	12.313	|0.219|	13.210|	11.916	|0.053|	3.105|
|32		|0.666|	8.654	|0.227|	9.550	|8.115	|0.096|	4.295|
|64		|0.628|	7.307	|0.246|	8.190	|6.478	|0.184|	5.009|
|128	|0.628|	8.668	|0.261|	9.570	|6.905	|0.227|	4.286|

![Total time on different blockSize](https://i.imgur.com/R9Lq9Nd.png)
![Speedup on different blockSize](https://i.imgur.com/tOcKPP5.png)

## Experience & conclusion

### What have you learned from this homework?

這次作業我學習了如何寫sse指令集的加速以及如何使用clang來編譯出auto vectorized的程式碼，也發現到要盡量將for loop裡的工作越早做越好，減少重複做的機會，另外也了解到程式中如果有if的話會降低效能，雖然有的時候有些if降低的效能還好，但是我以後會盡量試試看把能移除的if盡量去除，避免資源消耗。
