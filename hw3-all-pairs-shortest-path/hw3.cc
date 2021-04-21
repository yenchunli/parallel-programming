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
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r,       r,      0,                  r,              1);
        cal(B, r,       r,  r + 1,      round - r - 1,              1);
        cal(B, r,       0,      r,                  1,              r);
        cal(B, r,   r + 1,      r,                  1,  round - r - 1);

        /* Phase 3*/
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

    int RB = Round * B;
    int RBB = (Round+1) * B;

    if(RBB > n) RBB = n;
    
    #pragma omp parallel for schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {

#ifdef TIME
    double omp_t1 = omp_get_wtime();
#endif

        int block_internal_start_x = b_i * B;
        int block_internal_end_x = (b_i + 1) * B;
        block_internal_end_x =  (block_internal_end_x > n)? n : block_internal_end_x;
        
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;

            //if (block_internal_end_x > n) block_internal_end_x = n;
            //if (block_internal_end_y > n) block_internal_end_y = n;
            
            block_internal_end_y =  (block_internal_end_y > n)? n : block_internal_end_y;

            for (int k = RB; k < RBB; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                
                
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    int IK = Dist[i][k];
                    #pragma clang loop vectorize(enable) interleave(enable)
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
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
