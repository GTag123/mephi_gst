#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "compute_cpu.h"
#include "compute_gpu.h"
const float cpu_percent = 0.04;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 1) {
        fprintf(stderr, "Error: The number of MPI processes must be greater than 1.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }   

    long long total_size;
    long long cpu_total_size;
    long long gpu_total_size;

    unsigned char *data = NULL;
    unsigned char *local_data = NULL;

    if (rank == 0) {
        FILE *fp = fopen("data.bin", "rb");
        if (fp == NULL) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fseek(fp, 0, SEEK_END);
        total_size = ftell(fp);
        rewind(fp);

        if (total_size % 16 != 0) {
            printf("\ntotal_size: %lli\n", total_size);
            fprintf(stdout, "Error: File size %% 16 != 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ((long long)(total_size * cpu_percent) % 16 != 0) {
            printf("\total_size * cpu_percent: %lli\n", total_size * cpu_percent);
            fprintf(stdout, "Error: total_size * cpu_percent size %% 16 != 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }


        data = (unsigned char *)malloc(total_size);
        if (data == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t bytes_read = fread(data, 1, total_size, fp);
        assert(bytes_read == total_size);
        fclose(fp);

        cpu_total_size = (long long) (total_size * cpu_percent);
        gpu_total_size = total_size - cpu_total_size;
        cudaDeviceReset();
    }



    MPI_Bcast(&total_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cpu_total_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gpu_total_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int *sendcounts = (int *) malloc(size * sizeof(int));
    int *displs = (int *) malloc(size * sizeof(int));
    int base_count = ( (cpu_total_size / 16) / (size - 1) ) * 16;
    int remainder = ( (cpu_total_size / 16) % (size - 1) ) * 16;

    assert(remainder % 16 == 0);

    sendcounts[0] = gpu_total_size;
    displs[0] = 0;

    for (int i = 1; i < size; i++) {
        sendcounts[i] = base_count;
        if (i == size - 1 && remainder != 0) {
            sendcounts[i] += remainder;
        }
        displs[i] = (i > 0) ? displs[i - 1] + sendcounts[i - 1] : 0;
    }


    local_data = (unsigned char *)malloc(sendcounts[rank]);
    if (local_data == NULL) {
        perror("memory alloc error");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    double start_time = MPI_Wtime();
    MPI_Scatterv(data, sendcounts, displs, MPI_UNSIGNED_CHAR, local_data, sendcounts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        gpu_compute(local_data, sendcounts[rank]);
    } else {
        cpu_compute(local_data, sendcounts[rank]);
    }

    MPI_Gatherv(local_data, sendcounts[rank], MPI_UNSIGNED_CHAR, data, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;


    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        // stresstest
        // FILE *fp_out = fopen("output.bin", "wb");
        // if (fp_out == NULL) {
        //     perror("Error opening output file");
        //     MPI_Abort(MPI_COMM_WORLD, 1);
        // }
        //
        // fwrite(data, 1, total_size, fp_out);
        // fclose(fp_out);


        free(data);
        FILE *info_file = fopen("info.bin", "wb");
        if (info_file == NULL) {
            perror("Error opening info file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fwrite(&total_size, sizeof(total_size), 1, info_file);
        fwrite(&max_time, sizeof(max_time), 1, info_file);
        fclose(info_file);
        // printf("%lld:%.6f\n", total_size, max_time);
    }

    free(sendcounts);
    free(displs);
    free(local_data);

    MPI_Finalize();
    return 0;
}
