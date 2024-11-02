import subprocess

import numpy as np
import os
import gc
import struct


def generate_matrices(count: int) -> np.ndarray:
    return np.random.randint(0, 256, (count, 4, 4), dtype=np.uint8)


def get_transposed_matrices(matrices: np.ndarray) -> np.ndarray:
    if not matrices.flags['C_CONTIGUOUS']:
        raise RuntimeError("matrices != C_CONTIGUOUS")
    return np.transpose(matrices, axes=(0, 2, 1))


def get_byte_array_from_matrices(matrices: np.ndarray) -> bytes:
    return matrices.tobytes()


def save_matrices_to_binary_file(matrices: np.ndarray, filename: str):
    byte_data = get_byte_array_from_matrices(get_transposed_matrices(matrices))
    with open(filename, 'wb') as f:
        f.write(byte_data)


def get_info() -> str:
    with open("info.bin", "rb") as info_file:
        data = info_file.read()

    total_size = struct.unpack('q', data[0:8])[0]
    assert total_size % 16 == 0
    max_time = struct.unpack('d', data[8:16])[0]
    return f"{total_size // 16}:{max_time:.6f}"


def stress_test(tread_count: int, m_count: int):
    count = m_count  # 1024*1024/16
    matrices = generate_matrices(count)
    save_matrices_to_binary_file(matrices, 'data.bin')

    del matrices
    gc.collect()

    subprocess.run(["./main"], check=True)
    print(get_info())

    os.remove("data.bin")
    os.remove("info.bin")
    gc.collect()

if __name__ == '__main__':
    print("1 thread")
    tread_count = 1
    stress_test(tread_count, 10000)
    stress_test(tread_count, 100000)
    stress_test(tread_count, 1000000)
    stress_test(tread_count, 10000000)
    stress_test(tread_count, 100000000)
    stress_test(tread_count, 300000000)

    # print("2 thread")
    # tread_count = 2
    # stress_test(tread_count, 10000)
    # stress_test(tread_count, 100000)
    # stress_test(tread_count, 1000000)
    # stress_test(tread_count, 10000000)
    # stress_test(tread_count, 100000000)

    # print("4 thread")
    # tread_count = 4
    # stress_test(tread_count, 10000)
    # stress_test(tread_count, 100000)
    # stress_test(tread_count, 1000000)
    # stress_test(tread_count, 10000000)
    # stress_test(tread_count, 100000000)

    # print("8 thread")
    # tread_count = 8
    # stress_test(tread_count, 10000)
    # stress_test(tread_count, 100000)
    # stress_test(tread_count, 1000000)
    # stress_test(tread_count, 10000000)
    # stress_test(tread_count, 100000000)

    # print("16 thread")
    # tread_count = 16
    # stress_test(tread_count, 10000)
    # stress_test(tread_count, 100000)
    # stress_test(tread_count, 1000000)
    # stress_test(tread_count, 10000000)
    # stress_test(tread_count, 100000000)

    # print("32 thread")
    # tread_count = 32
    # stress_test(tread_count, 10000)
    # stress_test(tread_count, 100000)
    # stress_test(tread_count, 1000000)
    # stress_test(tread_count, 10000000)
    # stress_test(tread_count, 100000000)
