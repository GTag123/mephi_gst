import socket
import struct
import time
import numpy as np
import gc


class Client:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port

    def send_data_for_computation(self, data, retries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 0))
            sock.sendall(struct.pack('q', len(data) * retries))  # len(data) only in bytes!
            for _ in range(retries):
                sock.sendall(data)

            ack = struct.unpack('B', sock.recv(1))[0]
            # print("send status: " + str(ack))
            if ack != 2:
                raise RuntimeError("server dont send ack")

            # print(f"Data sent for computation.")

    def check_computation_result(self) -> (bytearray, float):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 1))
            result_size_data = sock.recv(8)
            result_size = struct.unpack('q', result_size_data)[0]

            if result_size == -1:
                sock.sendall(struct.pack('B', 2))
                # print("Computation is still in progress.")
            else:
                computation_time = struct.unpack('d', sock.recv(8))[0]
                sock.sendall(struct.pack('B', 2))
                return b'echo', computation_time
        return None, None

    def set_stress_byte(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack('B', 2))
            ack = struct.unpack('B', sock.recv(1))[0]
            # print("send status: " + str(ack))
            if ack != 2:
                raise RuntimeError("server dont send ack")
            # print(f"Stress byte set.")


def generate_matrices(count: int) -> np.ndarray:
    return np.random.randint(0, 256, (count, 4, 4), dtype=np.uint8)


def get_byte_array_from_matrices(matrices: np.ndarray) -> bytes:
    return matrices.tobytes()


def base_main():
    base_exp = 7
    min_exp = 5
    multiplier = int(input("Multiplier, default 1: ") or 1)
    exponent = int(input(f"Exponent, min - {min_exp}, default - {base_exp}: ") or base_exp)
    if exponent < min_exp:
        exponent = min_exp
    count = multiplier * 10 ** exponent
    batch_size = 10000

    print(f"{count} matrices")
    assert count % batch_size == 0

    client = Client()
    client.set_stress_byte()
    matrices = generate_matrices(batch_size)
    client.send_data_for_computation(get_byte_array_from_matrices(matrices), count // batch_size)
    gc.collect()
    while True:
        result, computation_time = client.check_computation_result()
        if result is not None:
            print(f"Computation time: {computation_time} s.")
            break
        time.sleep(2)
    client.set_stress_byte()

if __name__ == '__main__':
    for i in range(5):
        count = 10000 * (10 ** i)
        batch_size = 10000
        assert count % batch_size == 0

        client = Client()
        client.set_stress_byte()
        matrices = generate_matrices(batch_size)
        client.send_data_for_computation(get_byte_array_from_matrices(matrices), count // batch_size)
        gc.collect()
        while True:
            result, computation_time = client.check_computation_result()
            if result is not None:
                print(f"{count}:{computation_time}")
                break
            time.sleep(2)
        client.set_stress_byte()



