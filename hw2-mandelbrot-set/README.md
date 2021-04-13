# Parallel Programming HW2

---
106062209 Yen-Chun Li
---

## Pthread

我的方法是每一個thread平均的分攤圖片的高。舉例來說，如果一個圖片是800*600，max_thread_num 是7的話，則我們會做出會讓thread_id=0負責處理高度為(0,7,14,21,28...588)的像素，thread_id=1負責處理高度為(1,8,15,22,29...589)的像素。透過這樣的分配可以用一個較為簡單的方式達成均勻非配的效果。

---

### 自定義資料結構

struct MandelbrotArg 是一些傳遞到pthread裡的參數。

```c
typedef struct {
    int iters;
    double left;
    double right;
    double lower; 
    double upper; 
    int width;
    int height;
    int j_start;
    int j_end;
}MandelbrotArg;
```

- j_start: 高開始的值
- j_end: 高結束的值

---

### 自定義函數

- `calc_repeated` 是負責計算repeated的值的，使用指標是為了能直接修改原本的資料。
- `calc_mandelbrot` 是 pthread_create時傳遞的function pointer，負責mandelbrot運算的初始與計算
- `write_png` 負責寫入圖片

```c
void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0);
void *calc_mandelbrot(void *argv);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);
```

---

### calc_mandelbrot

```c
void *calc_mandelbrot(void *argv) {
    #ifdef TIME
        t1 = std::chrono::steady_clock::now();
    #endif
    MandelbrotArg *arg = (MandelbrotArg*) argv;
    int iters = arg->iters;
    double left = arg->left;
    double right = arg->right;
    double lower = arg->lower;
    double upper = arg->upper;
    int width = arg->width;
    int height = arg->height;
    int j_start = arg->j_start;  // thread_id
    int j_end = arg->j_end;      // 圖片的height
    
    // 提出一些常用的運算
    double tmp1 = (upper - lower) / height;
    double tmp2 = (right - left) / width;
    
    // 每一次處理間隔為num_of_threads的圖片
    for (int j = j_start; j < j_end; j+=num_of_threads) {
        double y0 = j * tmp1 + lower;
        int tmp3 = j * width;
        
        // 先計算x0，因為她可以獨立做計算
        double x0[width];
        #pragma GCC ivdep
        for(int i=0; i<width; ++i){
            x0[i] = i * tmp2 + left;
        }
        
        int i;
        // 計算repeats，這裡我們一次取兩個pixel出來做計算
        for(i=1; i < width; i+=2){
            double x[2] = {0};
            double y[2] = {0};
            double x_tmp[2] = {0};
            double y_tmp[2] = {0};
            double xx[2] = {0};
            double yy[2] = {0};
            double length_squared[2] = {0};
            double x0_arr[2] = {x0[i-1], x0[i]};
            int repeats=0;
            int state = 0; // 紀錄狀態
            // 如果兩個pixel都符合條件的話，就繼續計算且兩個pixel的repeats是一樣的
            // 如果不符合條件的話就會跳出while loop
            while(1){
                // 設定變數state，讓之後的程式知道是因為哪一個條件跳出while loop的
                if(length_squared[0] >= 4) { state = 1; break; }
                if(length_squared[1] >= 4) { state = 2; break; }
                if(repeats >= iters)       { state = 3; break; }
                
                // 多設兩個x_tmp, y_tmp減少dependency
                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k){
                    y_tmp[k] = 2 * x[k] * y[k] + y0;
                    x_tmp[k] = xx[k] - yy[k] + x0_arr[k];
                }

                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k){
                    y[k] = y_tmp[k];
                    x[k] = x_tmp[k];
                }
                
                // 多設兩個xx, yy做平方的預先處理，給下面的用
                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k){
                    xx[k] = x[k] * x[k];
                    yy[k] = y[k] * y[k];
                }
                
                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k)
                    length_squared[k] = xx[k] + yy[k];
                                            
                ++repeats;
            }

            // 儲存前後兩個pixel的index，讓下面使用
            int index1 = tmp3 + i - 1;
            int index2 = tmp3 + i;
            
            switch(state){
                // pixel[0]還沒做完
                case 2:
                    image[index2] = repeats;
                    while (length_squared[0] < 4 && repeats < iters) {
                        calc_repeated(&x[0], &y[0], &xx[0], &yy[0], &length_squared[0], &y0, &x0_arr[0]);
                        ++repeats;
                    }
                    
                    image[index1] = repeats;
                    break;
                // pixel[1]還沒做完
                case 1:
                    image[index1] = repeats;
                    while (length_squared[1] < 4 && repeats < iters) {
                        calc_repeated(&x[1], &y[1], &xx[1], &yy[1], &length_squared[1], &y0, &x0_arr[1]);
                        ++repeats;
                    }
                    image[index2] = repeats;
                    break;
                // 都做完
                case 3:
                    image[index1] = repeats;
                    image[index2] = repeats;
                    break;
                default:
                    break;
            }
        }
        // 最後一個pixel有可能沒有處理到
        if(( i = i-1) <width){
            int repeats = 0;
            
            double x = 0;
            double y = 0;

            double xx = 0;
            double yy = 0;
            
            double length_squared = 0;
            while (length_squared < 4 && repeats < iters) {
                calc_repeated(&x, &y, &xx, &yy, &length_squared, &y0, &x0[i]);
                ++repeats;
            }
            image[tmp3 + i] = repeats;
        }
        
    }
    #ifdef TIME
        t2 = std::chrono::steady_clock::now();
        std::cout << "[Thread " << j_start << "] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us.\n";
    #endif

    return NULL;
```

---

### calc_repeated

透過傳遞指標來直接更改值。

```c
void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0) {
    *y = 2 * *x * *y + *y0;
    *x = *xx - *yy + *x0;
    *xx = *x * *x;
    *yy = *y * *y;
    *length_squared = *xx + *yy;
}
```

---

### 程式初始化

初始一個pthread_t的陣列，長度為總共可以使用的核心數量。
初始一個MandelbrotArg的陣列，用於給thread初始條件

```c
pthread_t threads[num_of_threads];

MandelbrotArg args[num_of_threads];
```

---

### Pthread創建、傳遞參數並開始執行

```c
for(i = 0; i < num_of_threads; i++) {
    args[i].iters = iters;
    args[i].left = left;
    args[i].right = right;
    args[i].lower = lower;
    args[i].upper = upper;
    args[i].width = width;
    args[i].height = height;

    args[i].j_start = i;
    args[i].j_end = height;

    rc = pthread_create(&threads[i], NULL, calc_mandelbrot, (void*)&(args[i]));
    if(rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }
}
```

---

### Pthread join 等待所有thread執行完畢並寫入至圖片

```c    
for(i = 0; i < num_of_threads; i++) {
    pthread_join(threads[i], NULL);
}

/* draw and cleanup */
write_png(filename, iters, width, height, image);
free(image);

```

---

## Hybrid

一樣是透過圖片的高來平均的分配任務。跟pthread比較不同的地方在於，我們必須將每一個process處理完的圖片聚集起來，並依照當初分配的方式來重新組合成原本的圖片。我是使用 MPI_Gatherv 來進行此操作。

---

### 全域變數

為了方便計算定義了下面的全域變數。

```c
int rank;
int size;
int iters;
double left;
double right;
double lower; 
double upper; 
int width;
int height;
int r;         // 記錄height%num_of_threads
int d;         // 記錄height/num_of_threads
int *image;
```

---

### 自定義函數

基本上與pthread版本大致相同，除了`calc_mandelbrot`的型態不一樣(void* --> void)。

```c
void calc_mandelbrot(int* line_arr, int line_counter);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);
void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0);
```

---

### 主程式初始化

使用 omp_set_num_threads 設定 omp使用的thread數量。

```c
cpu_set_t cpu_set;
sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
//printf("%d cpus available\n", CPU_COUNT(&cpu_set));
int num_of_threads = CPU_COUNT(&cpu_set);
omp_set_num_threads(num_of_threads);
```

---

### 主程式分配任務給process

計算每個process應該要被分配到的height，然後存到line_arr，並將總共的數量存到line_counter。執行calc_mandelbrot開始計算。
```c
r = height % size;
d = height / size; 

int line_counter = (rank < r)?d+1:d;
int line_arr[line_counter];

#pragma omp parallel for
for(int i=0; i < line_counter; ++i)
    line_arr[i] = rank+i*size;

image = (int*)malloc(width * line_counter * sizeof(int));
assert(image);

calc_mandelbrot(line_arr, line_counter);
```

---

### 使用MPI_Gatherv從不同process蒐集處理好的資料

```c
int* image_buf = (int*)malloc(width * height * sizeof(int));

int* displs = (int*)malloc(size * sizeof(int));
int* recvcounts = (int*)malloc(size * sizeof(int));

displs[0] = 0;

for(int i=0; i < size; i++) {
    if(i < r) recvcounts[i] = (d+1) * width;
    else recvcounts[i] = d * width;
    if(i>=1) displs[i] = displs[i-1] + recvcounts[i-1];
}

MPI_Gatherv(image, width * line_counter, MPI_INT, image_buf, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
```

---

### 將分散於每個process中的pixel寫入到out.png

我有使用prama omp parallel for來加速，只是寫入的動作並不是主要的bottleneck，減少的時間有限。

```c

// Write to png
if(rank == 0) {
    #ifdef TIME
        t1 = std::chrono::steady_clock::now();
    #endif
    int* final_image = (int*)malloc(width * height * sizeof(int));

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(int k = 0; k < size; k++){
            for(int i = k, counter = 0; i < height; i+=size, ++counter){
                int l_index = i*width;
                int r_index = displs[k] + counter*width;


                #pragma omp simd
                for(int j=0; j<width; j++){
                    final_image[l_index+j] = image_buf[r_index + j];
                }

                //memcpy(&final_image[l_index], &image_buf[r_index], sizeof(int)*width);
            }
        }
    }




    write_png(filename, iters, width, height, final_image);


}

free(displs);
free(recvcounts);
free(image_buf);
/* draw and cleanup */

free(image);
MPI_Finalize();
return 0;

}
```

---

### void calc_mandelbrot(int* line_arr, int line_counter)

基本上跟pthread version很相近，只是在for loop 加上了#pragma omp parallel 和 #pragma omp for schedule(dynamic)來做加速。

---

## 實驗分析

我是選用strict35來做實驗，並把iterator的數量調高至50000，避免時間差異太小。
測量的方法我有使用到下列幾個方法:

1. t2 = std::chrono::steady_clock::now();

用於pthread算時間，還有算整個total process所需要的時間。

2. omp_timer[rank][omp_get_thread_num()] += omp_get_wtime() - omp_t1;

負責計算OMP裡面的thread個別執行了多久。

3. comm_time += MPI_Wtime() - time1;

計算MPI的溝通時間。

---

### Experiment 1: Pthread version when n=1

透過表格和下面的圖表可以發現，在可以使用的core越多的情況下，程式執行的速度是加快的，基本上多一倍的core，程式執行速度就加快1倍。但是我們可以發現在c=8時有分配不均勻的情況。

|  n  |       c  | Total Time| Average Thread time | Standard deviation | Speed up  |
| --- | -------- | --------- | ------------------- | ------------------ | --------- |
|   1 |        2 |   501.900 |              498.523|              0.587 |     1.000 |
|   1 |        4 |   252.650 |              249.168|              0.125 |     2.001 |   
|   1 |        6 |   169.650 |              166.147|              0.134 |     3.000 |
|   1 |        8 |   136.833 |              127.162|              3.782 |     3.920 |
|   1 |       10 |   102.900 |               99.654|              0.106 |     5.003 | 
|   1 |       12 |    86.083 |               83.120|              0.104 |     5.998 |

![Thread Time, pthread, n=1](https://i.imgur.com/PVjLGjZ.png)
![Speedup, pthread, n=1](https://i.imgur.com/QRtv2Ib.png)

---

### Experiment 2: Hybrid version when c=4


透過表格和下面的圖表可以發現，主要時間基本上都花在computation上面，觀察speedup也可以看到，基本上也是process開的越多，速度幾乎是1倍的加速。跟pthread版比較的話可以發現，hybrid的標準差幾乎都比較小。

|n|c|rearrage(s)|write Image(s)|Total Time(s)|Thread Average Time(s)|STDEV|Speedup|
|-|-|-|-|-|-|-|-|
| 2|4|0.0368|2.5927|	207.630|203.7312|0.0140|1.000|
| 4|4|0.0224|2.5799|	105.630|101.9169|0.0630|1.999|
| 6|4|0.0264|2.5817|	71.7000| 67.9396|0.0745|2.999|
| 8|4|0.0267|2.5765|	67.5800| 50.9315|0.0445|4.000|
|10|4|0.0242|2.5792|	44.4500| 40.7715|0.0191|4.997|
|12|4|0.0219|2.6059|	37.7000| 33.9459|0.0281|6.002|

![Total Time, Hybrid, c=4](https://i.imgur.com/C80Uw4q.png)
![Speedup, Hybrid c=4](https://i.imgur.com/1w5PAEl.png)

---

### Experiment 3: Hybrid version when n=4

透過表格及圖表可以觀察到，基本上結論跟Experiment 2類似，也可以觀察到(n=4, c=2) 跟(n=2, c=4)兩種情況的執行時間基本上是相近的。

|n|c|rearrage(s)|write Image(s)|Total Time(s)|Process Average Time(s)|STDEV|Speedup|
|-|-|-|-|-|-|-|-|-|-|-|
|4|	1	|0.063	|2.576|	411.033|407.561|0.133| 1.000|
|4|	2	|0.036	|2.577|	207.083|203.671|0.082| 2.001|
|4|	4	|0.022	|2.580|	105.533|101.891|0.033| 4.000|
|4|	6	|0.022	|2.579|	 71.550| 67.942|0.054| 5.999|
|4|	8	|0.022	|2.581|	 62.167| 58.258|0.027| 6.996|
|4|	10	|0.022	|2.578|	 44.317| 40.783|0.026| 9.993|
|4|	12	|0.021	|2.585|	 37.583| 33.975|0.017|11.996|


![Total Time, Hybrid, n=4](https://i.imgur.com/LeFNJdo.png)
![Speedup, Hybrid, n=4](https://i.imgur.com/CDqfgDX.png)

---

## 問題與討論

1. Scalability
這次兩支程式都可以有蠻好的擴展性，增加thread以後都可以有更好的效能。
2. Load Balance
Pthread版並沒有很平均的分散任務，有的時候會有一些core負擔比較重的任務，但是Hybrid版的Load balance還不錯，每一個process執行的時間蠻接近，推測應該是OpenMP分配的很均勻或是執行的次數還不夠大。

---

## 這次實驗學習到的東西

### 1. 利用ifdef和Makefile, 編譯特定程式碼

首先我們在sample.c中加入下面幾行，當TIME有被設定時，ifdef裡面的程式才會被編譯。
```c
//sample.c
#ifdef TIME
    printf("time=%d", time1);
#endif
```

---

接著我們修改Makefile，下面程式碼的意思是會去判斷TIME變數是否等於1，如果成立就會在CFLAGS新增TIME，讓上面的sample.c的程式碼可以偵測。
```bash
ifeq ($(TIME),1)
	CFLAGS += -DTIME
endif
```

---

然後我們就可以選擇要不要編譯含有輸出時間的程式

```bash
#要時間輸出
make TIME=1
#不要時間輸出
make
```

---

### 2. 學習到了如何vectorize

這次學習到了如何使用vectorize運算來加快程式，加上vectorize以後可以使程式幾乎快上一半，我這次作業是使用auto-vectorize來加速，原本是想要試看看AVX，但是因為apollo沒有支援到AVX指令集，最高只有SSE4.2而已所以沒有試成功。

### 3. 學習到如何使用omp快速加速程式碼

這次作業有使用到openmp，覺得很方便，能夠快速的將重複的工作用multi-thread來執行加速。現在試過最快的omp應該是schedule(dynamic)，因為它會動態的去安排工作給先執行完的thread，所以可以節省許多時間。

```c
#pragma omp parallel
{
    #pragma omp for schedule(dynamic)
    // ...
}
```


