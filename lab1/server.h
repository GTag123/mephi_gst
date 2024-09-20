#pragma once
#include <atomic>
#include <string>

constexpr unsigned char ack_code = 2;
constexpr int not_ready_code = -1;

void LOG(const std::string &message);

class Server {
public:
    explicit Server(int port) : port_(port), has_result_(false), raw_data_(nullptr), raw_data_size_(0) {
    }

    void start();

private:
    int port_;
    std::mutex compute_mutex_;
    std::atomic<bool> has_result_;
    unsigned char* raw_data_;
    int raw_data_size_;

    void handle_client(int client_socket);

    void send_computation_status(int client_socket);

    void perform_computation();

    static void send_ack(int client_socket);

    static void wait_ack(int client_socket);

    void clear_data();
};
