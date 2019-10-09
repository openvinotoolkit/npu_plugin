cmake_minimum_required(VERSION 3.5.0)
project(convertTensor VERSION 0.0.1)

find_package(gflags REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(convert_tensor 
    "convert_tensor.cpp"
)
target_link_libraries(convert_tensor gflags nlohmann_json::nlohmann_json)

