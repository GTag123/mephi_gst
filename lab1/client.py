import random
import socket
import struct
import time
from typing import List


class Client:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port

    def send_data_for_computation(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 0))
            sock.sendall(struct.pack('i', len(data)))  # len(data) only in bytes!
            sock.sendall(struct.pack('B' * len(data), *data))

            ack = struct.unpack('B', sock.recv(1))[0]
            print("send status: " + str(ack))
            if ack != 2:
                raise RuntimeError("server dont send ack")

            print(f"Data sent for computation.")

    def check_computation_result(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 1))
            result_size_data = sock.recv(4)
            result_size = struct.unpack('i', result_size_data)[0]

            if result_size == -1:
                sock.sendall(struct.pack('B', 2))
                print("Computation is still in progress.")
            else:
                result = bytearray()
                for _ in range(result_size):
                    result += sock.recv(1)
                sock.sendall(struct.pack('B', 2))
                return result


def generate_matrices(count: int) -> List[List[List[int]]]:
    matrices = []
    for _ in range(count):
        matrix = []
        for _ in range(4):
            row = []
            for _ in range(4):
                row.append(random.randint(0, 255))
            matrix.append(row)
        matrices.append(matrix)
    return matrices


def get_transposed_matrices(matrices: List[List[List[int]]]) -> List[List[List[int]]]:
    result = []
    for matrix in matrices:
        transposed_matrix = []
        for i in range(len(matrix)):
            transposed_matrix.append([matrix[j][i] for j in range(len(matrix))])
        result.append(transposed_matrix)
    return result


def get_byte_array_from_matrices(matrices: List[List[List[int]]]) -> bytearray:
    result = bytearray()
    for matrix in matrices:
        for row in matrix:
            for value in row:
                result.append(value)
    return result


def get_matrices_from_byte_array(data: bytearray) -> List[List[List[int]]]:
    size = len(data) // 16
    matrices = []
    for i in range(size):
        matrix = []
        for j in range(4):
            row = []
            for k in range(4):
                row.append(data[i * 16 + j * 4 + k])
            matrix.append(row)
        matrices.append(matrix)
    return matrices


def compute_in_server(matrices: List[List[List[int]]]) -> List[List[List[int]]]:
    data_to_send = get_byte_array_from_matrices(get_transposed_matrices(matrices))
    client = Client()
    client.send_data_for_computation(data_to_send)
    while True:
        result = client.check_computation_result()
        if result is not None:
            return get_transposed_matrices(get_matrices_from_byte_array(result))
        time.sleep(2)


def matrices_to_file(matrices: List[List[List[int]]], filename: str):
    with open(filename, 'w') as file:
        for matrix in matrices:
            for row in matrix:
                for value in row:
                    file.write(str(hex(value)) + ' ')
                file.write('\n')
            file.write('\n')


def get_date() -> str:
    return time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())


if __name__ == '__main__':
    matrices_in = generate_matrices(10000000)
    # matrices_to_file(matrices_in, 'files/matrices_in' + get_date() + '.txt')

    matrices_out = compute_in_server(matrices_in)
    # matrices_to_file(matrices_out, 'files/matrices_out' + get_date() + '.txt')
