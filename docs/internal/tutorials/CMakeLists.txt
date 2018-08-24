cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

add_executable(01_model_basics 01_model_basics.cpp)
target_sources(01_model_basics PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${tensor})

add_executable(02_op_model_composition 02_op_model_composition.cpp)
target_sources(02_op_model_composition PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${tensor})

add_executable(03_op_model_traversal 03_op_model_traversal.cpp)
target_sources(03_op_model_traversal PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${tensor})