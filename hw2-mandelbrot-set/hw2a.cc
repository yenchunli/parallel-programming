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

#include <chrono>
#include <iostream>
#include <pthread.h>

int* image;
unsigned int num_of_threads;

#ifdef TIME
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
#endif

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

void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0);
void *calc_mandelbrot(void *argv);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_of_threads = CPU_COUNT(&cpu_set);
    //printf("%d cpus available\n", num_of_threads);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[num_of_threads];

    MandelbrotArg args[num_of_threads];
    int rc;
    int i; 

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
    
    for(i = 0; i < num_of_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0) {
    *y = 2 * *x * *y + *y0;
    *x = *xx - *yy + *x0;
    *xx = *x * *x;
    *yy = *y * *y;
    *length_squared = *xx + *yy;
}

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
    int j_start = arg->j_start;
    int j_end = arg->j_end;
    /* mandelbrot set */

    double tmp1 = (upper - lower) / height;
    double tmp2 = (right - left) / width;
    
    for (int j = j_start; j < j_end; j+=num_of_threads) {
        double y0 = j * tmp1 + lower;
        int tmp3 = j * width;
        double x0[width];

        #pragma GCC ivdep
        for(int i=0; i<width; ++i){
            x0[i] = i * tmp2 + left;
        }
        
        int i;
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
            // two pixels executes
            int state = 0;
            while(1){
                if(length_squared[0] >= 4) { state = 1; break; }
                if(length_squared[1] >= 4) { state = 2; break; }
                if(repeats >= iters)       { state = 3; break; }

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
        
        
    }
    #ifdef TIME
        t2 = std::chrono::steady_clock::now();
        std::cout << "[Thread " << j_start << "] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us.\n";
    #endif

    return NULL;

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
            //printf("[%d, %d] = %d\n", y, x, buffer[y*width + x]);
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
