#pragma once

struct Computation {
    signed long long data_size;
    unsigned char* data;
    double computed_time;
};

// struct ComputationData {
//     int n;
//     unsigned char* raw_data; // хранит транспонированные матрицы
// };

// ComputitionData unpack_raw(unsigned char* raw_data, int size);
//
// void pack_raw(const ComputitionData& data, unsigned char* raw_data);
//
// void delete_data(const ComputitionData& data);