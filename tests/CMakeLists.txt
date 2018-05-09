cmake_minimum_required(VERSION 3.0.0)
project(fathom VERSION 1.0.0)

enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#add_executable(graph_container graph_container.cpp ${graph})
#include_directories(include/fathom/graph)

add_executable(MCM_unit_tests MCM_unit_tests.cpp graph_container_test.cpp graph_visualizer_test.cpp model_serializer_test.cpp)
target_link_libraries (MCM_unit_tests gtest)
target_sources(MCM_unit_tests PUBLIC ${allocator})

set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES stdc++)
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
set_target_properties(MCM_unit_tests PROPERTIES LINKER_LANGUAGE C)
