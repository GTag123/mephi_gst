cmake_minimum_required(VERSION 3.28)
project(lab1)

set(CMAKE_CXX_STANDARD 20)

add_executable(lab1 main.cpp
        server.cpp
        server.h
        compute.cpp
        compute.h
        utils.cpp
        utils.h)
