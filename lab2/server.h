#pragma once
#include <atomic>
#include <string>
#include "utils.h"

constexpr unsigned char ack_code = 2;
constexpr signed long long not_ready_code = -1;

void LOG(const std::string &message);

class Server {
public:
    explicit Server(int port) : port_(port), has_result_(false) {
        computation_.computed_time = 0;
        computation_.data = nullptr;
        computation_.data_size = 0;
    }

    void start();

private:
    int port_;
    std::mutex compute_mutex_;
    std::atomic<bool> has_result_;
    std::atomic<bool> is_stress_test_;
    Computation computation_{};

    void handle_client(int client_socket);

    void send_computation_status(int client_socket);

    void perform_computation();

    static void send_ack(int client_socket);

    static void wait_ack(int client_socket);

    void clear_data();
    void clear_computation();
};
