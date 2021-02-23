#include <cstdio>

#include <mpi.h>

#include <boost/sort/spreadsort/spreadsort.hpp>

#define TAG1 1
#define TAG2 2
#define TAG3 3
#define TAG4 4
#define TAG5 5
#define TAG6 6

void mergeArray(float * arr1, float * arr2, unsigned int * length1, unsigned int * length2, float * arr3);

unsigned int calculateDataLength(int total_process, unsigned int * array_length, int rank);

int main(int argc, char ** argv) {

    if (argc != 4) {
        fprintf(stderr, "Must provide 3 parameters\n");
        return -1;
    }

    MPI_Init( & argc, & argv);

    unsigned int array_length = atoll(argv[1]);
    char * input_filename = argv[2];
    char * output_filename = argv[3];

    int rank, total_process;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank); // the rank (id) of the calling process
    MPI_Comm_size(MPI_COMM_WORLD, & total_process); // the total number of processes

    MPI_File f;
    MPI_File f2;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, & f);
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, & f2);

    unsigned int data_length = calculateDataLength(total_process, & array_length, rank);

    int max_rank;
    if (total_process > array_length) max_rank = array_length - 1;
    else max_rank = total_process - 1;

    int sum_of_sorted = 0;
    int sorted = 0;

    float * data;
    unsigned int start_idx;

    unsigned int rr;
    unsigned int dd;
    unsigned int ddd;
    float * new_data;
    float * tmp_data;
    bool isEven;
    if (rank % 2 == 0) isEven = true;
    else isEven = false;
    int flag;
    int flag_buf;

    if (data_length != 0) {

        rr = array_length % total_process;
        dd = array_length / total_process;
        ddd = dd + 1;
        data = (float * ) malloc(sizeof(float) * ddd);
        new_data = (float * ) malloc(sizeof(float) * ddd);
        tmp_data = (float * ) malloc(sizeof(float) * (ddd + dd));

        if (rank <= rr) start_idx = ddd * rank;
        else start_idx = ddd * rr + (rank - rr) * dd;

        MPI_File_read_at(f, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);

        boost::sort::spreadsort::spreadsort(data, data + data_length);
    }

    MPI_Request request1;
    MPI_Request request2;

    MPI_Status status;

    if (total_process == 1) {
        MPI_File_write_at(f2, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
        free(data);
        MPI_Finalize();
        return 0;
    }

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
            if (isEven && rank != 0) {
                MPI_Isend(data, data_length, MPI_FLOAT, rank - 1, TAG3, MPI_COMM_WORLD, & request1);
            }

            if (!isEven && rank != max_rank) {

                unsigned int new_data_length = calculateDataLength(total_process, & array_length, rank + 1);

                MPI_Recv(new_data, new_data_length, MPI_FLOAT, rank + 1, TAG3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (data[data_length - 1] > new_data[0]) {
                    sorted = 0;
                    unsigned int tmp_data_length = data_length + new_data_length;

                    // merge two array and sort
                    mergeArray(data, new_data, & data_length, & new_data_length, tmp_data);

                    for (int i = 0; i < data_length; ++i) {
                        data[i] = tmp_data[i];
                    }

                    flag = 1;

                    MPI_Isend( & flag, 1, MPI_INT, rank + 1, TAG6, MPI_COMM_WORLD, & request1);

                    MPI_Isend( & tmp_data[data_length], new_data_length, MPI_FLOAT, rank + 1, TAG4, MPI_COMM_WORLD, & request2);

                } else {

                    flag = 0;

                    MPI_Isend( & flag, 1, MPI_INT, rank + 1, TAG6, MPI_COMM_WORLD, & request1);

                }

            }

            if (isEven && rank != 0) {

                MPI_Recv( & flag_buf, 1, MPI_INT, rank - 1, TAG6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (flag_buf == 1) {

                    MPI_Recv(data, data_length, MPI_FLOAT, rank - 1, TAG4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                }

            }

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
}

void mergeArray(float * arr1, float * arr2, unsigned int * length1, unsigned int * length2, float * arr3) {
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < * length1 && j < * length2) {
        if (arr1[i] < arr2[j]) arr3[k++] = arr1[i++];
        else arr3[k++] = arr2[j++];
    }
    while (i < * length1)
        arr3[k++] = arr1[i++];

    while (j < * length2)
        arr3[k++] = arr2[j++];
}

unsigned int calculateDataLength(int total_process, unsigned int * array_length, int rank) {
    // array length:  5
    // total process: 13
    if (total_process >= * array_length) {
        if (rank < * array_length) return 1;
        else return 0;
    }
    // array length: 13
    // total process: 5
    else {
        unsigned int rest = * array_length % total_process;
        unsigned int data_length = * array_length / total_process;
        if (rank < rest) return data_length + 1;
        else return data_length;
    }
}