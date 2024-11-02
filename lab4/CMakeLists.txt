cmake_minimum_required(VERSION 3.25)
project(openmpi)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)

add_executable(openmpi compute.c)

target_link_libraries(openmpi PRIVATE MPI::MPI_C)