#include "compute.h"
#include <cstdint>
#include <stdio.h>
#include <time.h>

uint8_t galois_multiply_by_2(const uint8_t value) {
    return (value << 1) ^ ((value & 0x80) ? 0x1B : 0x00);
}

uint8_t galois_multiply_by_3(const uint8_t value) {
    return galois_multiply_by_2(value) ^ value;
}

void mix_single_column(uint8_t *column) {
    uint8_t temp[4];

    temp[0] = galois_multiply_by_2(column[0]) ^ galois_multiply_by_3(column[1]) ^ column[2] ^ column[3];
    temp[1] = column[0] ^ galois_multiply_by_2(column[1]) ^ galois_multiply_by_3(column[2]) ^ column[3];
    temp[2] = column[0] ^ column[1] ^ galois_multiply_by_2(column[2]) ^ galois_multiply_by_3(column[3]);
    temp[3] = galois_multiply_by_3(column[0]) ^ column[1] ^ column[2] ^ galois_multiply_by_2(column[3]);

    for (int i = 0; i < 4; i++) {
        column[i] = temp[i];
    }
}

void mix_columns(uint8_t **state) {
    for (int i = 0; i < 4; i++) {
        mix_single_column(state[i]);
    }
}

void compute(const ComputitionData data) {
    clock_t start = clock();
    for (int i = 0; i < data.n; i++) {
        mix_columns(data.matrices[i]);
    }
    clock_t end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Время выполнения: %.6f секунд\n", cpu_time_used);
}
