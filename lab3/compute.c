#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

unsigned char galois_multiply_by_2(const unsigned char value) {
    return (value << 1) ^ ((value & 0x80) ? 0x1B : 0x00);
}

unsigned char galois_multiply_by_3(const unsigned char value) {
    return galois_multiply_by_2(value) ^ value;
}

void mix_single_column(unsigned char *column) {
    unsigned char temp[4];

    temp[0] = galois_multiply_by_2(column[0]) ^ galois_multiply_by_3(column[1]) ^ column[2] ^ column[3];
    temp[1] = column[0] ^ galois_multiply_by_2(column[1]) ^ galois_multiply_by_3(column[2]) ^ column[3];
    temp[2] = column[0] ^ column[1] ^ galois_multiply_by_2(column[2]) ^ galois_multiply_by_3(column[3]);
    temp[3] = galois_multiply_by_3(column[0]) ^ column[1] ^ column[2] ^ galois_multiply_by_2(column[3]);

    for (int i = 0; i < 4; i++) {
        column[i] = temp[i];
    }
}

void mix_columns(unsigned char *matrix) {
    for (int i = 0; i < 4; i++) {
        mix_single_column(matrix + i * 4);
    }
}

void compute(unsigned char *data, const long long size) {
    // clock_t start = clock();
    assert(data != NULL);
    assert(size % 16 == 0);
    for (long long i = 0; i < size / 16; i++) {
        mix_columns(data + i * 16);
    }
    // clock_t end = clock();
    // printf("Время выполнения: %.6f секунд\n", ((double) (end - start)) / CLOCKS_PER_SEC);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    long long total_size;
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


        data = (unsigned char *)malloc(total_size);
        if (data == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t bytes_read = fread(data, 1, total_size, fp);
        assert(bytes_read == total_size);
        fclose(fp);
    }



    MPI_Bcast(&total_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int base_count = ( (total_size / 16) / size ) * 16;
    int remainder = ( (total_size / 16) % size ) * 16;

    assert(remainder % 16 == 0);

    for (int i = 0; i < size; i++) {
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
    compute(local_data, sendcounts[rank]);
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