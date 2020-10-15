cmake_minimum_required(VERSION 3.5.0)
project(graphComparator VERSION 0.0.1)

include_directories(${PROJECT_SOURCE_DIR}/tools/graph_comparator)
include_directories(${PROJECT_SOURCE_DIR}/contrib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(
    graph_comparator_tests
    graph_comparator_tests.cpp
)

target_link_libraries(graph_comparator_tests pthread gtest gcomparator)
