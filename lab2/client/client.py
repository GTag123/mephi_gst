import socket
import struct
import time
import numpy as np
import gc

class Client:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port

    def send_data_for_computation(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 0))
            sock.sendall(struct.pack('q', len(data)))  # len(data) only in bytes!
            sock.sendall(data)

            ack = struct.unpack('B', sock.recv(1))[0]
            print("send status: " + str(ack))
            if ack != 2:
                raise RuntimeError("server dont send ack")

            print(f"Data sent for computation.")

    def check_computation_result(self) -> (bytearray, float):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 1))
            result_size_data = sock.recv(8)
            result_size = struct.unpack('q', result_size_data)[0]

            if result_size == -1:
                sock.sendall(struct.pack('B', 2))
                print("Computation is still in progress.")
            else:
                computation_time = struct.unpack('d', sock.recv(8))[0]

                result = bytearray(result_size - 8)
                view = memoryview(result)
                total_received = 0
                while total_received < len(result):
                    chunk = sock.recv(len(result) - total_received)
                    if not chunk:
                        raise ConnectionError()
                    view[total_received:total_received + len(chunk)] = chunk
                    total_received += len(chunk)

                sock.sendall(struct.pack('B', 2))
                return result, computation_time
        return None, None


def generate_matrices(count: int) -> np.ndarray:
    return np.random.randint(0, 256, (count, 4, 4), dtype=np.uint8)


def get_transposed_matrices(matrices: np.ndarray) -> np.ndarray:
    if not matrices.flags['C_CONTIGUOUS']:
        # matrices = np.ascontiguousarray(matrices)
        raise RuntimeError("matrices != C_CONTIGUOUS")
    return np.transpose(matrices, axes=(0, 2, 1))


def get_byte_array_from_matrices(matrices: np.ndarray) -> bytes:
    return matrices.tobytes()


def get_matrices_from_byte_array(data: bytearray) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, 4, 4)


def compute_in_server(matrices: np.ndarray) -> np.ndarray:
    client = Client()
    client.send_data_for_computation(get_byte_array_from_matrices(get_transposed_matrices(matrices)))

    del matrices
    gc.collect()

    while True:
        result, computation_time = client.check_computation_result()
        if result is not None:
            print(f"Computation time: {computation_time} s.")
            return get_transposed_matrices(get_matrices_from_byte_array(result))
        time.sleep(2)


def matrices_to_file(matrices: np.ndarray, filename: str):
    with open(filename, 'w') as file:
        for matrix in matrices:
            for row in matrix:
                file.write(' '.join([str(value) for value in row]) + '\n')
            file.write('\n')


def get_date() -> str:
    return time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())


if __name__ == '__main__':
    # matrices_in_per_mb = 65536  # 1024*1024/16
    # matrices_in = generate_matrices(int(input("MB: ")) * matrices_in_per_mb)

    matrices_in = generate_matrices(int(input("N: ")))
    matrices_to_file(matrices_in, 'files/matrices_in' + get_date() + '.txt')

    matrices_out = compute_in_server(matrices_in)
    matrices_to_file(matrices_out, 'files/matrices_out' + get_date() + '.txt')
