cmake_minimum_required(VERSION 3.0.0)
project(mcm VERSION 1.0.0)

enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#add_executable(graph_container graph_container.cpp ${graph})
#include_directories(include/mcm/graph)

add_executable(mcm_unit_tests 
    mcm_unit_tests.cpp 
    graph_container_test.cpp
    graph_visualizer_test.cpp
    base_components_test.cpp
    ops_test.cpp
    cm_composition_test.cpp
    cm_data_model_test.cpp
    model_serializer_test.cpp)
target_link_libraries (mcm_unit_tests gtest m)
target_sources(mcm_unit_tests PUBLIC ${allocator} ${computation_model} ${logger})

set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES stdc++)
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
set_target_properties(mcm_unit_tests PROPERTIES LINKER_LANGUAGE C)
