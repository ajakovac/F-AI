cmake_minimum_required (VERSION 3.0)
project (F-AI)

# do not forget to set export CXX=/usr/bin/g++-10
set(CXX "g++-10")
set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -Wall -Wextra")

# for more files in the project: add_library

add_executable(gputest gputest.cpp)

target_link_libraries(gputest -lOpenCL)


