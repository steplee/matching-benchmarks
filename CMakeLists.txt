cmake_minimum_required(VERSION 3.9)

project(grid LANGUAGES CXX CUDA)

add_definitions(-std=c++11 -O3)

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS timer system)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIRECTORY}/src)

#set(CMAKE_CUDA_FLAGS "--use_fast_math")

add_library(grid_impl
  #src/grid_cpu.cc
  src/grid_cpu_naive.cc
  src/grid_cpu_naive_mt.cc
  #src/grid_cpu_mt.cc
  src/grid_cuda_naive.cu
  #src/grid_cuda.cu
)

add_executable(grid_bench src/bench.cc)

target_link_libraries(grid_impl PUBLIC ${Boost_TIMER_LIBRARY} ${Boost_SYSTEM_LIBRARY})
target_link_libraries(grid_bench grid_impl)
