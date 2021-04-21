#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <math.h>

int rank;
int size;
int iters;
double left;
double right;
double lower; 
double upper; 
int width;
int height;

int r;
int d;
int *image;

#ifdef TIME
    double time1;
    double comm_time = 0;
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    double omp_timer[12][12];
#endif

void calc_mandelbrot(int* line_arr, int line_counter);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);
void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0);

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int num_of_threads = CPU_COUNT(&cpu_set);
    omp_set_num_threads(num_of_threads);
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    // MPI Init
    MPI_Init(&argc, &argv);
    #ifdef TIME
        time1 = MPI_Wtime();
    #endif
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    #ifdef TIME
        comm_time += MPI_Wtime() - time1;
    #endif
    //printf("rank=%d, size=%d\n", rank, size);

    r = height % size;
    d = height / size; 
    
    int line_counter = (rank < r)?d+1:d;

    int line_arr[line_counter];

    #pragma omp parallel for
    for(int i=0; i < line_counter; ++i) {
        line_arr[i] = rank+i*size;
    }

    /* allocate memory for image */
    image = (int*)malloc(width * line_counter * sizeof(int));
    assert(image);
    
    #ifdef TIME
        t1 = std::chrono::steady_clock::now();
    #endif
    
    calc_mandelbrot(line_arr, line_counter);
    
    #ifdef TIME
        t2 = std::chrono::steady_clock::now();
        std::cout << "[Rank=" << rank << "]" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us.\n";
        

        for(int i=0; i<num_of_threads; ++i)
            printf("[Rank=%d][thread_id=%d]%f\n", rank, i, omp_timer[rank][i]);
    #endif
    
    int* image_buf = (int*)malloc(width * height * sizeof(int));
    
    int* displs = (int*)malloc(size * sizeof(int));
    int* recvcounts = (int*)malloc(size * sizeof(int));

    displs[0] = 0;

    for(int i=0; i < size; i++) {
        if(i < r) recvcounts[i] = (d+1) * width;
        else recvcounts[i] = d * width;
        if(i>=1) displs[i] = displs[i-1] + recvcounts[i-1];
    }

    /*
    for(int i=0; i < size; i++) {
        printf("%d ", displs[i]);
    }
    printf("\ndispls\n");
    for(int i=0; i < size; i++) {
        printf("%d ", recvcounts[i]);
    }
    printf("\nrecvcounts\n");
    */
    #ifdef TIME
        time1 = MPI_Wtime();
    #endif
    MPI_Gatherv(image, width * line_counter, MPI_INT, image_buf, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    #ifdef TIME
        comm_time += MPI_Wtime() - time1;
    #endif
    

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
        

        #ifdef TIME
            t2 = std::chrono::steady_clock::now();
            std::cout << "[Rearrange]" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us.\n";
            
            t1 = std::chrono::steady_clock::now();
        #endif

        write_png(filename, iters, width, height, final_image);
        
        #ifdef TIME
            t2 = std::chrono::steady_clock::now();
            std::cout << "[WriteImage]" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us.\n";

            free(final_image);

            printf("[Comm Time] %f", comm_time);
        #endif
    }

    free(displs);
    free(recvcounts);
    free(image_buf);
    /* draw and cleanup */
    
    free(image);
    MPI_Finalize();
	return 0;
    
}

void calc_mandelbrot(int* line_arr, int line_counter) {

    //printf("rank=%d, j_start=%d, j_end=%d, size=%d\n", rank, j_start, j_end, size);
    /* mandelbrot set */

    double tmp1 = (upper - lower) / height;
    double tmp2 = (right - left) / width;

    //printf("line_counter=%d\n", line_counter);
    //printf("rank=%d ", rank);
    //for(int i=0; i<line_counter;i++)    printf("%d ", line_arr[i]);
    //printf("hello=%d", line_arr[line_counter-1]);
    //printf("\n");
    //int counter = 0;
    #pragma omp parallel 
    {
        #pragma omp for schedule(dynamic) 
        for (int j = 0; j < line_counter; ++j) {
            
            #ifdef TIME
                double omp_t1 = omp_get_wtime();
            #endif

            int jdx = line_arr[j];
            double y0 = jdx * tmp1 + lower;
            int tmp3 = j * width;
            double x0[width];

            #pragma simd
            for(int i=0; i<width; ++i){
                x0[i] = i * tmp2 + left;
            }
            
            int i;
            int v_size = 2;
            for(i=v_size-1; i < width; i+=v_size){
                double x[v_size] = {0};
                double y[v_size] = {0};
                double x_tmp[v_size] = {0};
                double y_tmp[v_size] = {0};
                double xx[v_size] = {0};
                double yy[v_size] = {0};
                double length_squared[v_size] = {0};
                double x0_arr[v_size] = {x0[i-1], x0[i]};
                int repeats=0;
                int state = 0;
                // two pixels executes
                while(1){
                    if(length_squared[0] >= 4) { state = 1; break; }
                    if(length_squared[1] >= 4) { state = 2; break; }
                    if(repeats >= iters)       { state = 3; break; }
                    
                    #pragma simd vectorlength (2)
                    for(int k=0; k < v_size; ++k){
                        
                        y_tmp[k] = 2 * x[k] * y[k] + y0;
                        x_tmp[k] = xx[k] - yy[k] + x0_arr[k];
                    }
                    
                    #pragma simd vectorlength (2)
                    for(int k=0; k < v_size; ++k){
                        
                        y[k] = y_tmp[k];
                        x[k] = x_tmp[k];
                    }

                    #pragma simd vectorlength (2)
                    for(int k=0; k < v_size; ++k){

                        xx[k] = x[k]*x[k];
                        yy[k] = y[k]*y[k];
                    }
                    
                    #pragma simd vectorlength (2)
                    for(int k=0; k < v_size; ++k){
                        length_squared[k] = xx[k] + yy[k];
                                               
                    }
                    ++repeats;
                }

                int index1 = tmp3 + i - 1;
                int index2 = tmp3 + i;

                switch(state){
                    case 2:
                        image[index2] = repeats;
                        while (length_squared[0] < 4 && repeats < iters) {
                            calc_repeated(&x[0], &y[0], &xx[0], &yy[0], &length_squared[0], &y0, &x0_arr[0]);
                            ++repeats;
                        }
                        
                        image[index1] = repeats;
                        break;
                    case 1:
                        image[index1] = repeats;
                        while (length_squared[1] < 4 && repeats < iters) {
                            calc_repeated(&x[1], &y[1], &xx[1], &yy[1], &length_squared[1], &y0, &x0_arr[1]);
                            ++repeats;
                        }
                        image[index2] = repeats;
                        break;
                    case 3:
                        image[index1] = repeats;
                        image[index2] = repeats;
                        break;
                    default:
                        break;
                }
            }

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
            #ifdef TIME
                omp_timer[rank][omp_get_thread_num()] += omp_get_wtime() - omp_t1;
            #endif
        }
    }
    
    
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0) {
    *y = 2 * *x * *y + *y0;
    *x = *xx - *yy + *x0;
    *xx = pow(*x, 2);
    *yy = pow(*y, 2);
    *length_squared = *xx + *yy;
}
