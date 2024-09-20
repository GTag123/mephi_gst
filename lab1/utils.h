#pragma once

struct ComputitionData {
    int n;
    unsigned char*** matrices; // хранит транспонированные матрицы
};

ComputitionData unpack_raw(const unsigned char* raw_data, int size);

void pack_raw(const ComputitionData& data, unsigned char* raw_data);

void delete_data(const ComputitionData& data);