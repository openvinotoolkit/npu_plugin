cmake_minimum_required(VERSION 3.5.0)
project(graphComparator VERSION 0.0.1)

include_directories(${PROJECT_SOURCE_DIR}/meta)
include_directories(${PROJECT_SOURCE_DIR}/contrib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(graph_comparator 
    "main.cpp"
    "src/graph_comparator.cpp"
    ${base} 
    ${logger}
)

add_library(gcomparator STATIC
    "src/graph_comparator.cpp"
    ${base} 
    ${logger}
)

target_link_libraries(graph_comparator pthread gtest)
add_dependencies(graph_comparator generate-graphfile-headers)
add_dependencies(gcomparator generate-graphfile-headers)

add_subdirectory(tests)