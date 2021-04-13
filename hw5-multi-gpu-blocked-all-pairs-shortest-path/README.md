# Blocked All-Pairs Shortest Path (Multi-cards)

Yen-Chun Li



## Implementation

### How do you divide your data?

First, I add the additional data to make the length of data is divided by 64. For example, if the total metrics is 82, we will add the additional data to make its length become 64*2 = **128**.

I choose 64 as my blocking factors.

The configuration of my code is in Table.1:

| Kernel function | Grid Size             | Block Size |
| --------------- | --------------------- | ---------- |
| phase1          |     1                 | (32, 32)   |
| phase2          | (Round, 2)            | (32, 32)   |
| phase3          | (Round, Round)        | (32, 32)   |

> Table 1. The configuration in phase1,2,3

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

### How do you implement the communication?

主要是在phase3時，將圖片分成上下兩區塊，紅色的部分是給GPU0做，藍色的部分是給GPU1做。
block floyd warshal運算的時候有dependency的部分只有Pivot-row跟pivot-column的部分。
但是透過下圖可以看出來，我們可以只要傳pivot-row就可以了，
因為pivot column的部分，之前傳的pivot-row有傳過了，超過pivot-row的部分也還不會被算到，
會在phase2的時候更新pivot-column的值。

```c
    *          //有更動
    *          //有更動
* * * * * * *  //有更動
    *          //沒更動
    *          //沒更動
    *          //沒更動
    *          //沒更動
```
我是利用OpenMP產生兩條thread分別控制兩張顯卡，將pivot row的部分在GPU之間傳送，我是使用CudaMemcpyPeer來做GPU之前的Communication。



![](https://i.imgur.com/cQaUpHK.png)

![](https://i.imgur.com/Nn3nXtm.png)




## Experiment & Analysis

### System Spec

hades.cs.nthu.edu.tw

### Weak Scalability

Observe weak scalability of the mulit-GPU implementations.

### Time Distribution

Analyze the time spent in:

1. computing
2. communication
3. memory copy (H2D, D2H)
4. I/O of your program w.r.t. input size.

|Testcase|v|Problem Size|Total(s)|Input(s)|Output(s)|Compute(s)|
|-|-|-|-|-|-|-|
|c01|5	    |    125	    |0.531330	|0.300358	|0.000056	|0.230916
|c02|160	|   4096000	    |0.468061	|0.255587	|0.000153	|0.212321
|c03|999	|  97002999	    |0.508910	|0.282370	|0.005140	|0.221400
|c04|5000	|1.25E+11	    |1.060658	|0.688902	|0.044676	|0.327080
|c05|11000	|1.331E+12	    |1.716411	|0.434437	|0.185830	|1.096144
|p16|16000	|4.096E+12	    |3.951392	|1.204183	|0.026570	|2.720639
|p21|20959	|9.20686E+12	|6.493136	|0.920829	|0.026806	|5.545501
|p26|25889	|1.73519E+13	|11.101869  |1.180990	|0.033938	|9.886941
|p31|31000	|2.9791E+13	    |20.570848  |2.400624	|1.380707	|16.789517
|p34|34921	|4.25853E+13	|26.761774  |2.764130	|0.045377	|23.952267
|c06|39857	|6.33161E+13	|38.578996  |2.847176	|2.268196	|33.463624
|c07|44939	|9.07549E+13	|53.879104  |3.154683	|2.881724	|47.842697

> ![Total Time](https://i.imgur.com/H6DQqDk.png)
> Fig. 1 Total time when problem size grows.
 
Fig.1 可以看到我的程式隨著Problem Size增加時，時間幾乎也是等比例增加的，這說明我的程式有良好的Scalability。

> ![GPU Time Distribution](https://i.imgur.com/9lToRDD.png)
> Fig.2 GPU Time Distribution

Fig.2 可以發現到Computation Time隨著node增加而增加，我們也可以看到Communication Time也從一開始沒有占很多時間，隨著Size變大後也占了蠻大一部分時間。

> ![GPU Time Distribution 2](https://i.imgur.com/vCk1tZz.png)
> Fig.3 GPU Time Distribution Percentage

Fig.3 可以看到各項時間所占的總百分比大概是多少，我們可以看到隨著size增加，communication Time的比重有上升的趨勢，Computation Time則有略微減少，這告訴我們說在vertex數量很多的時候，我們必須去考慮多個GPU所需要的溝通時間是否能做優化或是降低。

## Experience & conclusion

這次實驗讓我學習到了如何使用多卡GPU來進行運算，需要對原本的演算法進行適當的調整不然雙卡的速度甚至會比單卡還慢，大部分時間都花在無謂的溝通上面。

## Images

![](https://i.imgur.com/rySFMik.png)

![](https://i.imgur.com/6QGInp7.png)

![](https://i.imgur.com/MBEa5id.png)

![](https://i.imgur.com/CdmyLmv.png)

![](https://i.imgur.com/XydFEJb.png)

![](https://i.imgur.com/x8ej7aK.png)

![](https://i.imgur.com/3fjeyfr.png)

![](https://i.imgur.com/7N4e2fM.png)

![](https://i.imgur.com/Kr7ChVu.png)

![](https://i.imgur.com/RgGFxUk.png)

![](https://i.imgur.com/Nh9dM6f.png)

![](https://i.imgur.com/KtEVF9N.png)

![](https://i.imgur.com/Bk4LidX.png)



