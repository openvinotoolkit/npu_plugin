cmake_minimum_required(VERSION 3.5.0)
project(validateTensor VERSION 0.0.1)

add_subdirectory(contrib/gflags)

add_executable(validate 
    "validate.cpp"
)

target_link_libraries(validate gflags cm)
