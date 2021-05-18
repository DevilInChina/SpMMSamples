cmake_minimum_required(VERSION 3.16)
project(homework_4_SpMM CUDA C)
add_compile_options(-O3)
set(CMAKE_CUDA_STANDARD 14)
add_executable(homework_4_SpMM time.c my_time.h main.cu)


set_target_properties(
        homework_4_SpMM
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)