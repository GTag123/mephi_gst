#include "compute.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

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

void compute(unsigned char *data, const long long size, double *time) {
    clock_t start = clock();
    assert(size % 16 == 0);
    for (long long i = 0; i < size / 16; i++) {
        mix_columns(data + i * 16);
    }
    clock_t end = clock();

    *time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Время выполнения: %.6f секунд\n", *time);
}
