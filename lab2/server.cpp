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
                long long data_size;
                recv(client_socket, &data_size, sizeof(data_size), 0);
                LOG("DATA SIZE: " + std::to_string(data_size));

                has_result_.store(false);
                clear_computation();
                computation_.data_size = data_size;
                computation_.data = new unsigned char[data_size];

                if (computation_.data == nullptr) {
                    throw std::runtime_error("Failed to allocate memory");
                }
                ssize_t readed = 0;
                while (readed < data_size) {
                    ssize_t n = recv(client_socket, computation_.data + readed,
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
            } else if (request_type == 2) {
                is_stress_test_ = !is_stress_test_;
                send_ack(client_socket);
                LOG(std::format("Server stress test is: {}", is_stress_test_.load()));
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
        long long result_size = sizeof(computation_.computed_time);
        if (!is_stress_test_.load()) {
            result_size += computation_.data_size;
        }
        send(client_socket, &result_size, sizeof(result_size), 0);
        send(client_socket, &computation_.computed_time, sizeof(computation_.computed_time), 0);
        if (!is_stress_test_.load()) {
            send(client_socket, computation_.data, result_size, 0);
        }
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
    compute(computation_.data, computation_.data_size, &computation_.computed_time);
    if (is_stress_test_) {
        clear_data();
    }
    has_result_ = true;
}

void Server::clear_data() {
    if (computation_.data != nullptr) {
        delete[] computation_.data;
        computation_.data = nullptr;
        computation_.data_size = 0;
    }
}

void Server::clear_computation() {
    clear_data();
    computation_.computed_time = 0;
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
