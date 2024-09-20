#include "utils.h"

#include <cstring>
#include <stdexcept>

ComputitionData unpack_raw(const unsigned char *raw_data, const int size) {
    ComputitionData data{};
    if (size % 16 != 0) {
        throw std::runtime_error("Data is not aligned to 16 bytes");
    }
    data.n = size / 16;
    data.matrices = new unsigned char **[data.n];
    for (int i = 0; i < data.n; ++i) {
        data.matrices[i] = new unsigned char *[4];
        for (int j = 0; j < 4; ++j) {
            data.matrices[i][j] = new unsigned char[4];
            memcpy(data.matrices[i][j], raw_data + 16 * i + 4 * j, 4);
        }
    }
    return data;
}

void pack_raw(const ComputitionData& data, unsigned char* raw_data) {
    for (int i = 0; i < data.n; ++i) {
        for (int j = 0; j < 4; ++j) {
            memcpy(raw_data + 16 * i + 4 * j, data.matrices[i][j], 4);
        }
    }
}

void delete_data(const ComputitionData& data) {
    for (int i = 0; i < data.n; ++i) {
        for (int j = 0; j < 4; ++j) {
            delete[] data.matrices[i][j];
        }
        delete[] data.matrices[i];
    }
    delete[] data.matrices;
}