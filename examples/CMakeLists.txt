cmake_minimum_required(VERSION 3.0.0)
project(fathom VERSION 1.0.0)

add_executable(graph_container graph_container.cpp)
target_sources(graph_container PUBLIC ${allocator})

add_executable(computation_model computation_model.cpp)
target_sources(computation_model PUBLIC ${allocator} ${computation_model} ${logger} ${tensor})