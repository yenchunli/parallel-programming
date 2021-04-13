# PP_2020_HW1

name: Yen-Chun Li
student ID: 106062209

## Implementation

### How do you handle an arbitrary number of input items and processes?

我的想法如下:
1. 如果全部的process數目, 每一個process分一個直到分完。
> 例子: 如果Process數量為5, array長度為3，則分配方式為(1,1,1,0,0)。
2. 先平均分配，最後一輪會剩下無法平均分配的數量，這時一個process分配一個直到分完。
> 例子: 如果Process數量為5，array長度為13，則先計算`13/5 = 2`，每個process先分長度2，剩下`13%5=3` 份再一個個分給process直到分完，所以會變成(3,3,3,2,2)。

### How do you sort in your program?

我的想法如下:
1. 先對每個process內進行sorting，這邊使用的是boost函式庫裡的spreadsort。
2. Process A 裡的資料跟Process B裡的資料比對時是先比對A的資料末端是否小於B的資料前端，如果成立的話代表A最大的數字也比B最小的數字小，表示已經排序完成，就會傳一個SIGNAL告訴Process B說已經sort過了；如果不成立的話，就會將兩邊的data合併並同時由小到大排列，再將較大的那部分取原先Process B 資料的長度傳回Process B。

### 詳細步驟


#### 第一部分： 初始化
```c=
    if(argc != 4){
		fprintf(stderr, "Must provide 3 parameters\n");
		return -1;
	}

	MPI_Init(&argc,&argv);

	unsigned int array_length = atoll(argv[1]);
	char *input_filename = argv[2];
	char *output_filename = argv[3]; 
	
	int rank, total_process;	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the rank (id) of the calling process
	MPI_Comm_size(MPI_COMM_WORLD, &total_process); // the total number of processes

	MPI_File f;
	MPI_File f2;
	MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f2);

	unsigned int data_length = calculateDataLength(total_process, &array_length, rank);

	int max_rank;
	if(total_process > array_length) max_rank = array_length - 1;
	else max_rank = total_process - 1;
```
- 首先我們先檢查parameter的長度是否為4，如果不是4的話代表使用者輸入錯誤，跳出程式。
- `MPI_INIT()`初始MPI
- `array_length`的型態設為`unsigned int`，因為spec上說array length的值的範圍為`1 ≤ n ≤ 536870911`，大概為INT_MAX的四分之一，我們這邊使用unsigned int，unsigned int的速度理論上比int快。
- `MPI_File_open()` 使用了3個參數，MPI_MODE_RDONLY(讀)、MPI_MODE_CREATE(創建檔案)、MPI_MODE_WONLY(寫)
- 這邊使用自定義的calculateDataLength來計算出每一個rank需要處理的data長度。`calculateDatalength` 這個函式會根據process的rank、array length以及number of process來計算出每一個process應該要分配到多少長度的資料。方法是跟上面說的方式一樣，盡量平均分配到每一個process。

```c=
unsigned int calculateDataLength(int total_process, unsigned int *array_length, int rank) {
	// array length:  5
	// total process: 13
	if(total_process >= *array_length) {
		if(rank < *array_length) return 1;
		else return 0;
	}
	// array length: 13
	// total process: 5
	else {
		unsigned int rest = *array_length % total_process;
		unsigned int data_length = *array_length / total_process;
		if(rank < rest) return data_length + 1;
		else return data_length;
	}
}
```
- 計算出最大rank的index是多少，存到變數 `max_rank` 裡面。

#### 第二部分，讀取資料並預先排序

如果process的data length不為0的情況下，進行以下的步驟。

- `(float*)data` 儲存這個process被分配到的資料
- `(float*)new_data` 儲存這個process收到其他process的資料
- `(float*)tmp_data` 儲存兩個process資料合併時的暫存資料
- 計算讀取時的start index
    - 如果rank <= rr，代表是前面幾個多分配到1個array length的process，start_idx的值為(dd+1)*rank。
    - 如果rank > rr，代表是後面沒有多分配的process，start_idx = (dd+1) * rr + (rank-rr) * dd。
- MPI_File_read_at()會從指定的start index，讀取長度為data length的資料，這邊由於我們已經計算好不同的process應該要讀取的位置，所以process會讀取自己被分配到的部分
- 使用`boost::spreadsort`排序process分配到的data，透過預先排序讓之後能以較簡單的方式來計算是否已經排序了
```c=
if(data_length !=0){

    rr = array_length % total_process;
    dd = array_length / total_process;
    ddd = dd + 1;
    data = (float*)malloc(sizeof(float) * ddd);
    new_data = (float*)malloc(sizeof(float) * ddd);
    tmp_data = (float*)malloc(sizeof(float) * (ddd+dd));

    if(rank <= rr) start_idx = ddd * rank;
    else start_idx = ddd * rr + (rank - rr) * dd;

    MPI_File_read_at(f, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);

    boost::sort::spreadsort::spreadsort(data, data + data_length);
}
```

#### 第三部分： 如果只允許一個process的話，直接將sort過的結果寫入檔案並結束

```c=
if(total_process == 1) {
    MPI_File_write_at(f2, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
    free(data);
    MPI_Finalize();
    return 0;
}
```

#### 第四部分： ODD-EVEN Sort

- 這邊主要是一個while迴圈來進行ODD-EVEN Sort，我們定義了一個變數sum_of_sorted，表示已經sort好的process的數量總和
- 進入迴圈後先將變數sorted設為1，假設我們已經sort好，如果之後發現沒有sort好再將sorted設為0
- 接著判斷如果data_length != 0 時，才需要運算，當data_length = 0，代表該process沒有資料要處理，sorted直接等於1
- 如果rank是奇數的話
    - 將自己的data用`MPI_Isend()`傳送到rank是偶數的process。`MPI_Isend()`是non-blocking的API。
- 如果rank是偶數且不是最後一個process的話，才要進行處理。(因為最後一個process如果rank是偶數的話是沒有人會對他送資料)
    - 計算收到data的長度，存到變數new_data_length裡
    - 使用`MPI_Recv()`來接收資料`
- 如果本身的資料的最大值，大於收到資料的最小值時，代表還沒有sort好，進行以下步驟：
    - 將sorted設為0，表示還沒sort好
    - 使用`mergeArray()`將兩邊的資料由小排到大，合併存至tmp_data
    - 使用for迴圈將tmp_data裡從index=0開始長度為data_length的data存回data中
    - 使用`MPI_Isend()`，傳送一個flag告訴其他process說這次有重新sort過
    - 使用`MPI_Isend()`，傳送處理好的資料給其他process
- 如果本身的資料的最大值，小於收到資料的最小值，代表已經sort好了，傳送一個flag告訴其他process這次不用重新sort，使用原本的資料就好
- 如果rank是奇數的話，等待flag的回傳，如果flag=1，表示重新sort過，則要使用`MPI_Recv()`來接收重新sort後的資料。
- EVEN SORT的部分基本上跟ODD SORT差不多，主要差在奇數偶數Node的判斷
- 最後使用`MPI_Allreduce()`來將每一個process中的變數sorted用`MPI_SUM`算總和，如果總和等於process全部的數量，代表全部都已經排序完了，就可以跳出while loop，如果沒有的話，繼續進行while loop
- 跳出while loop後，先判斷data length是否為零，如果是零代表沒有資料要寫入;不為0的話，將自己分配到的data依照原本的start_idx寫入到FILE中，並釋放malloc的記憶體，執行`MPI_FINALIZE()`結束MPI程式

```c=
while (sum_of_sorted < total_process) {
    sorted = 1;
    if (data_length != 0) {
        if (!isEven) {
            MPI_Isend(data, data_length, MPI_FLOAT, rank - 1, TAG1, MPI_COMM_WORLD, & request1);
        }
        if (isEven && rank != max_rank) {
            unsigned int new_data_length = calculateDataLength(total_process, & array_length, rank + 1);
            MPI_Recv(new_data, new_data_length, MPI_FLOAT, rank + 1, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // if not sorted
            // merge two sort lists and return max one
            //
            if (data[data_length - 1] > new_data[0]) {
                sorted = 0;
                unsigned int tmp_data_length = data_length + new_data_length;
                // merge two array and sort
                mergeArray(data, new_data, & data_length, & new_data_length, tmp_data);
                for (int i = 0; i < data_length; ++i) {
                    data[i] = tmp_data[i];
                }
                flag = 1;
                MPI_Isend( & flag, 1, MPI_INT, rank + 1, TAG5, MPI_COMM_WORLD, & request1);
                MPI_Isend( & tmp_data[data_length], new_data_length, MPI_FLOAT, rank + 1, TAG2, MPI_COMM_WORLD, & request2);
            }
            // if sorted
            // send nothing to rank + 1
            else {
                flag = 0;
                MPI_Isend( & flag, 1, MPI_INT, rank + 1, TAG5, MPI_COMM_WORLD, & request1);
            }
        }
        if (!isEven) {
            MPI_Recv( & flag_buf, 1, MPI_INT, rank - 1, TAG5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (flag_buf == 1) {
                MPI_Recv(data, data_length, MPI_FLOAT, rank - 1, TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // EVEN SORT
        //..........
        //..........
         
    }
    MPI_Allreduce( & sorted, & sum_of_sorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

if (data_length != 0) {

    MPI_File_write_at(f2, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);

    free(data);
    free(new_data);
    free(tmp_data);
}

MPI_Finalize();

return 0;
```



## Experiment & Analysis 

### Methodology

#### System Spec (If you run your experiments on your own cluster)

我使用apollo.cs.nthu.edu.tw的機器來進行我的實驗。

#### Performance Metrics: How do you measure the computing time, communication time and IO time? How do you compute the values in the plots?

我使用MPI_Wtime()來包住要計算之程式碼的前後。以下面作為例子，MPI_File_open是一個IO動作，所以我們把它的時間加進變數io_time裡面。

我們在一開始宣告了變數 comm_time 和 io_time，分別儲存不同的時間。

```c=
double t1, t2, t3, t4, t5, t6, t7, t8;
double comm_time = 0;
double io_time = 0;

MPI_Init();

t1 = MPI_Wtime();
MPI_File f;
MPI_File f2;
MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f2);
t2 = MPI_Wtime();
io_time += (t2-t1);
```

![](https://i.imgur.com/PmS7aIi.png)

我們使用Sbatch遞交任務，生成Slurm檔案來分析LOG。LOG裡面會有每個PROCESS的執行時間，我們將他取平均當作是總共的時間。用總共的時間扣掉comm_time和io_time就會得到大概的CPU運算時間。

1-1.sh
```bash=
#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
echo "N1 n1"
srun time ./hw1 536869888 /home/pp20/share/hw1/testcases/35.in ./out
```

1-2.sh
```bash=
#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
echo "N1 n2"
srun time ./hw1 536869888 /home/pp20/share/hw1/testcases/35.in ./out
```

test.sh
```bash
sbatch 1-1.sh
sbatch 1-2.sh
```

```bash=
sh test.sh
```

### Plots: Speedup Factor & Time Profile

#### 情況1: 固定Node=1時，不同數目的process

Table. 1 列出了在固定node數量為1時，不同數量的process的效能比較。我們從Fig. 1可以發現到CPU Time是隨著process數量增加而降低的，而IO Time基本持平，COMM Time基本上是隨著process數量增加而增加，因為有更多的process需要互相通信。另外有觀察到，雖然process=8時的COMM TIME比process=4時還少，但是若以Fig.3的百分比來看的話，COMM TIME隨著process增加所佔的比例越來越吃重，符合我們的猜測。

|# of Nodes | # of process | Comm. Time | IO Time | CPU Time | Total Time |Speedup|
|-----------|--------------|------------|---------|----------|------------|-------|
|1|1|	0.00|	4.46|	26.18	|30.64|	1.00|
|1|2|	2.79|	3.41|	14.88	|21.08|	1.45|
|1|4|	5.92|	2.84|	10.20	|18.96|	1.62|
|1|8|	5.26|	2.36|	6.44	|14.05|	2.18|


Table 1. CPU, COMM, IO Time Under different number of process when node=1.

![](https://i.imgur.com/gtUiKY0.png)


Fig. 1 CPU, COMM, IO Time under different number of process when node=1.

![](https://i.imgur.com/3RcRFTw.png)

Fig. 2 Speedup under different number of process when node=1.

![](https://i.imgur.com/KXP2iD7.png)


Fig. 3 當node=1時，各個時間所佔的百分比


#### 情況2: 固定process下，不同數量的node

在固定`# of processes＝12`時，我們可以發現到總共的時間是隨著node數量的增加而減少，推測應該是分到多個node後，每一個process分到的運算資源較為足夠。

|# of Nodes | # of process | Comm. Time | IO Time | CPU Time | Total Time |Speedup|
|-----------|--------------|------------|---------|----------|------------|-------|
|1|12|6.70|2.21|6.71|15.61|1.00|
|2|12|6.51|1.97|6.69|15.17|1.03|
|3|12|6.02|2.03|6.53|14.58|1.07|
|4|12|5.43|2.09|6.39|13.91|1.12|

Table.2 當total process=12的Time Consumption。

![](https://i.imgur.com/q1Sqzej.png)

Fig.3  當total process=12，時間分布圖。

![](https://i.imgur.com/jfoEFmu.png)

Fig.4 當total process=12，Speed up的折線圖。




#### 情況3: 固定node下，不同數量的process

我們將node數量固定在4，觀察process的數量是否會影響時間。這裡我們發現到CPU Time基本上隨著process的增加而減少，但是同一時間我們也可以發現COMM Time所佔的比例越來越高。

|# of Nodes | # of process | Comm. Time | IO Time | CPU Time | Total Time |Speedup|
|-----------|--------------|------------|---------|----------|------------|-------|
|4|4	|5.58	|3.40	|8.99	|17.97|1.00|
|4|8	|5.55	|2.22	|6.96	|14.73|1.22|
|4|12	|5.66	|2.03	|6.26	|13.95|1.29|
|4|16	|6.69	|1.88	|5.98	|14.55|1.24|
|4|20	|6.58	|3.93	|5.83	|16.34|1.10|
|4|24	|6.37	|2.08	|5.00	|13.44|1.34|
|4|28	|6.33	|2.02	|5.13	|13.48|1.33|
|4|32	|6.56	|2.13	|4.63	|13.32|1.35|
|4|36	|6.54	|3.29	|5.40	|15.22|1.18|
|4|40	|6.64	|4.26	|5.14	|16.04|1.12|
|4|44	|6.45	|2.42	|4.23	|13.11|1.37|
|4|48	|6.79	|2.72	|4.54	|14.05|1.28|



![](https://i.imgur.com/tYd6Sy4.png)

![](https://i.imgur.com/5QvBRq1.png)

![](https://i.imgur.com/yPxMVpQ.png)


#### 情況4: 不同排序方式

我一開始是使用std::sort()，因為記得std::sort很快，但是後來上網路和問同學後，聽聞有一個boost library，可以很快的加速sorting。使用後讓我的程式加快很多，於是做了一個小實驗來比較不同sorting algorithm的速度。

|Sorting Algorithm| Comm. Time | IO Time | CPU Time | Total Time |
|-|-|-|-|-|-|
|boost::spread_sort	|5.42|1.96| 6.22|13.61|
|boost::pdqsort	    |5.40|1.92| 6.51|13.82|
|std::sort	        |5.41|1.88| 9.11|16.40|
|std::stable_sort	|5.44|1.88|10.00|17.32|

從圖表中可以發現，spread_sort是所有sorting algorithm裡面最快的。我們可以觀察到，較快的sorting algorithm可以節省相當多的CPU Time，來使得總體的速度顯著降低。

![Different algo.](https://i.imgur.com/1MFunVF.png)



### Discussion (Must base on the results in your plots)

#### Compare I/O, CPU, Network performance. Which is/are the bottleneck(s)? Why? How could it be improved?
在process數量很少的時候，CPU是主要的bottleneck，因為沒有什麼傳遞訊息的需求。
在process數量變多的時候，因為把工作都分散到許多個CPU來做運算，CPU的時間會慢慢降低到一個瓶頸就很難在降低了，這時候Network performance會變成主要的瓶頸，因為不同的Process之間需要溝通，網路會影響溝通傳遞的速度。然而，如果我們想降低COMM Time，就必須減少Process，但是這也會影響運算的時間，所以我們必須針對DATA的分布型態來找到最適合的process數量。



#### Compare scalability. Does your program scale well? Why or why not? How can you achieve better scalability? You may discuss for the two implementations separately or together.

我覺得我的程式的scalability不太好，增加的process數量並沒有很顯著的減少整體的時間。不過也有可能是實驗做的次數不夠，造成了誤差。不過基本上程式的CPU Time有隨著Process增加而下降，應該是有成功分散式運算吧。

### Others 

## Experiences / Conclusion
It can include these following aspects:
- Your conclusion of this assignment.
不是process數量越多，程式就會跑的越快。process太多的話會花很多時間在傳送接收資料。
好的sorting algorithm可以讓程式跑很快很快。

- What have you learned from this assignment?
這次的實驗讓我更加了解MPI的工作原理以及如何將程式跑在很多的process上。
- What difficulties did you encounter in this assignment?
我花了很多時間在優化程式碼上面，但是時間只有減少一點點，還有一開始使用MPI_IRecv()時會一直收不到封包，後來改用MPI_Recv就成功了。
- If you have any feedback, please write it here. Such as comments for improving the spec of this assignment, etc.
希望之後會有助教傳授小秘訣時間，想知道如何加速程式碼。

