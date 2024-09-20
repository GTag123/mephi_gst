#include "server.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <netinet/in.h>
#include "compute.h"

void LOG(const std::string &message) {
    const auto now = std::chrono::system_clock::now();
    const auto now_c = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::tm now_tm = *std::localtime(&now_c);
    std::cout << "[" << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << "."
            << std::setw(3) << std::setfill('0') << milliseconds.count() << "] "
            << message << std::endl;
}

void Server::start() {
    sockaddr_in server_addr{}, client_addr{};
    socklen_t client_addr_len = sizeof(client_addr);

    const int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        LOG("Failed to create socket");
        return;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);

    if (bind(server_socket, reinterpret_cast<sockaddr *>(&server_addr),
             sizeof(server_addr)) < 0) {
        LOG("Bind failed");
        return;
    }

    listen(server_socket, 5);
    LOG("Server listening on port " + std::to_string(port_));

    while (true) {
        int client_socket = accept(server_socket, reinterpret_cast<sockaddr *>(&client_addr),
                                   &client_addr_len);
        if (client_socket < 0) {
            LOG("Failed to accept connection");
            continue;
        }

        LOG("Client connected");
        std::thread(&Server::handle_client, this, client_socket).detach();
    }

    // close(server_socket);
}

void Server::handle_client(const int client_socket) {
    try {
        while (true) {
            unsigned char request_type;
            ssize_t read_size = recv(client_socket, &request_type, sizeof(request_type), 0);
            if (read_size <= 0) {
                LOG("Connection closed");
                close(client_socket);
                return;
            }

            if (request_type == 0) {
                int data_size;
                recv(client_socket, &data_size, sizeof(data_size), 0);
                LOG("DATA SIZE: " + std::to_string(data_size));

                has_result_.store(false);
                clear_data();
                raw_data_size_ = data_size;
                raw_data_ = new unsigned char[raw_data_size_];

                if (raw_data_ == nullptr) {
                    throw std::runtime_error("Failed to allocate memory");
                }
                ssize_t readed = 0;
                while (readed < data_size) {
                    ssize_t n = recv(client_socket, raw_data_ + readed,
                                     data_size - readed, 0);
                    if (n > 0) {
                        readed += n;
                    } else {
                        LOG(std::format("n:{} <= 0", n));
                        close(client_socket);
                        return;
                    }
                }

                std::thread(&Server::perform_computation, this).detach();

                send_ack(client_socket);
                LOG("Received data for computation");
            } else if (request_type == 1) {
                send_computation_status(client_socket);
            }
        }
    } catch (std::runtime_error &e) {
        LOG(std::format("Closing connection: {}", e.what()));
    }

    close(client_socket);
}


void Server::send_computation_status(const int client_socket) {
    if (has_result_.load()) {
        std::lock_guard lock(compute_mutex_);
        const int result_size = raw_data_size_;

        send(client_socket, &result_size, sizeof(result_size), 0);
        send(client_socket, raw_data_, result_size, 0);
        wait_ack(client_socket);

        LOG("Sent computation result to client");
    } else {
        send(client_socket, &not_ready_code, sizeof(not_ready_code), 0);
        wait_ack(client_socket);
        LOG("Computation not done yet");
    }
}

void Server::perform_computation() {
    has_result_ = false;
    std::lock_guard lock(compute_mutex_);
    const ComputitionData packed_data = unpack_raw(raw_data_, raw_data_size_);
    compute(packed_data);
    pack_raw(packed_data, raw_data_);
    delete_data(packed_data);

    has_result_ = true;
}

void Server::clear_data() {
    if (raw_data_ != nullptr) {
        delete[] raw_data_;
        raw_data_size_ = 0;
    }
}


void Server::send_ack(const int client_socket) {
    send(client_socket, &ack_code, sizeof(ack_code), 0);
}

void Server::wait_ack(const int client_socket) {
    unsigned char code = 0;
    ssize_t n = recv(client_socket, &code, sizeof(code), 0);
    if (n != 1 && code != 2) {
        throw std::runtime_error("client dont send ack for result");
    }
}
