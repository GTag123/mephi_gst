import subprocess

import numpy as np
import struct
import gc


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
    print(f"Matrices saved to binary file: {filename}")


def load_matrices_from_binary_file(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        byte_data = f.read()

    matrices = np.frombuffer(byte_data, dtype=np.uint8).reshape(-1, 4, 4)
    return np.transpose(matrices, axes=(0, 2, 1))


def save_matrices_to_text_file(matrices: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        for matrix in matrices:
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\n')
    print(f"Matrices saved to text file: {filename}")


if __name__ == '__main__':
    compute_cnt = int(input("compute_cnt: "))
    is_stress = bool(int(input("Is stress: ")))
    type = int(input("Type: "))
    count = 0
    if type == 1:
        count = int(input("N: "))
    elif type == 2:
        count = int(input("MB: ")) * 65536  # 1024*1024/16
    else:
        raise RuntimeError("incorrect type")

    matrices = generate_matrices(count)
    save_matrices_to_binary_file(matrices, 'data.bin')

    del matrices
    gc.collect()

    process = subprocess.run(['mpirun', '-np', f"{compute_cnt}", "./main"], check=True)

    if not is_stress:
        in_matrices = load_matrices_from_binary_file('data.bin')
        save_matrices_to_text_file(in_matrices, 'input.txt')

        matrices = load_matrices_from_binary_file('output.bin')
        save_matrices_to_text_file(matrices, 'output.txt')
