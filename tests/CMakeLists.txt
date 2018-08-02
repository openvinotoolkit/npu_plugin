cmake_minimum_required(VERSION 3.0.0)
project(mcm VERSION 1.0.0)

enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#add_executable(graph_container graph_container.cpp ${graph})
#include_directories(include/mcm/graph)

file(GLOB base_src
    "base/*.cpp"    
)

file(GLOB op_src
    "op/*.cpp"
)

file(GLOB pass_src
    "pass/*.cpp"
)

file(GLOB compiler_src
    "compiler/*.cpp"
)

file(GLOB model_src
    "model/*.cpp"
)

file(GLOB graph_src
    "graph/*.cpp"
)

add_executable(mcm_unit_tests 
    mcm_unit_tests.cpp 
    ${graph_src}
    ${base_src}
    ${op_src}
    ${pass_src}
    ${compiler_src}
    ${model_src}
)

target_link_libraries (mcm_unit_tests gtest m)
target_sources(mcm_unit_tests PUBLIC ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${base} ${tensor} ${compiler})

set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES stdc++)
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
set_target_properties(mcm_unit_tests PROPERTIES LINKER_LANGUAGE C)
