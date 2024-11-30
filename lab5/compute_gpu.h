__device__ unsigned char gpu_galois_multiply_by_2(const unsigned char value) {
    return (value << 1) ^ ((value & 0x80) ? 0x1B : 0x00);
}

__device__ unsigned char gpu_galois_multiply_by_3(const unsigned char value) {
    return gpu_galois_multiply_by_2(value) ^ value;
}

__device__ void gpu_mix_single_column(unsigned char *column) {
    unsigned char temp[4];

    temp[0] = gpu_galois_multiply_by_2(column[0]) ^ gpu_galois_multiply_by_3(column[1]) ^ column[2] ^ column[3];
    temp[1] = column[0] ^ gpu_galois_multiply_by_2(column[1]) ^ gpu_galois_multiply_by_3(column[2]) ^ column[3];
    temp[2] = column[0] ^ column[1] ^ gpu_galois_multiply_by_2(column[2]) ^ gpu_galois_multiply_by_3(column[3]);
    temp[3] = gpu_galois_multiply_by_3(column[0]) ^ column[1] ^ column[2] ^ gpu_galois_multiply_by_2(column[3]);

    for (int i = 0; i < 4; i++) {
        column[i] = temp[i];
    }
}

__global__ void gpu_mix_columns_kernel(unsigned char *data, long long num_blocks) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long step = gridDim.x * blockDim.x;


    for (long long i = idx; i < num_blocks; i += step) {
        if (i < num_blocks) {
            gpu_mix_single_column(data + i * 16);
        }
    }
}

void gpu_compute(unsigned char *data, const long long size) {
    unsigned char *d_data;
    long long num_blocks = size / 16;

    cudaMalloc((void**)&d_data, size * sizeof(unsigned char));
    cudaMemcpy(d_data, data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = 1024;

    gpu_mix_columns_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}