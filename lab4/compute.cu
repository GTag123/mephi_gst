#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ unsigned char galois_multiply_by_2(const unsigned char value) {
    return (value << 1) ^ ((value & 0x80) ? 0x1B : 0x00);
}

__device__ unsigned char galois_multiply_by_3(const unsigned char value) {
    return galois_multiply_by_2(value) ^ value;
}

__device__ void mix_single_column(unsigned char *column) {
    unsigned char temp[4];

    temp[0] = galois_multiply_by_2(column[0]) ^ galois_multiply_by_3(column[1]) ^ column[2] ^ column[3];
    temp[1] = column[0] ^ galois_multiply_by_2(column[1]) ^ galois_multiply_by_3(column[2]) ^ column[3];
    temp[2] = column[0] ^ column[1] ^ galois_multiply_by_2(column[2]) ^ galois_multiply_by_3(column[3]);
    temp[3] = galois_multiply_by_3(column[0]) ^ column[1] ^ column[2] ^ galois_multiply_by_2(column[3]);

    for (int i = 0; i < 4; i++) {
        column[i] = temp[i];
    }
}

__global__ void mix_columns_kernel(unsigned char *data, long long num_blocks) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long step = gridDim.x * blockDim.x;


    for (long long i = idx; i < num_blocks; i += step) {
        if (i < num_blocks) {
            mix_single_column(data + i * 16);
        }
    }
}

void compute(unsigned char *data, const long long size) {
    unsigned char *d_data;
    long long num_blocks = size / 16;

    cudaMalloc((void**)&d_data, size * sizeof(unsigned char));
    cudaMemcpy(d_data, data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = 1024;

    mix_columns_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

int main(int argc, char *argv[]) {
    FILE *fp = fopen("data.bin", "rb");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long long total_size = ftell(fp);
    rewind(fp);

    if (total_size % 16 != 0) {
        printf("Error: File size %% 16 != 0\n");
        fclose(fp);
        return 1;
    }

    unsigned char *data = (unsigned char *)malloc(total_size);
    if (data == NULL) {
        perror("Memory allocation error");
        fclose(fp);
        return 1;
    }

    size_t bytes_read = fread(data, 1, total_size, fp);
    assert(bytes_read == total_size);
    fclose(fp);

    clock_t start = clock();
    compute(data, total_size);
    clock_t end = clock();
    cudaDeviceReset();

    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    FILE *info_file = fopen("info.bin", "wb");
    if (info_file == NULL) {
        perror("Error opening info file");
        free(data);
        return 1;
    }
    fwrite(&total_size, sizeof(total_size), 1, info_file);
    fwrite(&elapsed_time, sizeof(elapsed_time), 1, info_file);
    fclose(info_file);

    // Запись данных (опционально)
    FILE *fp_out = fopen("output.bin", "wb");
    fwrite(data, 1, total_size, fp_out);
    fclose(fp_out);

    free(data);
    printf("Total size: %lld, Elapsed time: %.6f seconds\n", total_size, elapsed_time);

    return 0;
}
