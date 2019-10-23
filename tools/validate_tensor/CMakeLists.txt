cmake_minimum_required(VERSION 3.5.0)
project(validateTensor VERSION 0.0.1)

find_package(gflags REQUIRED)

add_subdirectory(contrib/nlohmann/json)

add_executable(validate 
    "validate.cpp"
)

target_link_libraries(validate gflags nlohmann_json::nlohmann_json corebase)

