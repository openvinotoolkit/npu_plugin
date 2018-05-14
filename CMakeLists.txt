cmake_minimum_required(VERSION 3.1.0)
project(fathom VERSION 1.0.0)

set (CMAKE_CXX_STANDARD 11)

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
include_directories(${PROJECT_SOURCE_DIR})
include_directories(include/fathom/graph)
include_directories(include/fathom/cm)

file(GLOB allocator "src/graph/base_allocator.cpp")
file(GLOB computation_model "src/computation/model/model.cpp" "src/computation/model/op_model.cpp" "src/computation/model/data_model.cpp" "src/computation/model/element.cpp" "src/computation/op/computation_op.cpp" "src/computation/op/kernel_op.cpp" "src/computation/model/attribute.cpp")
file(GLOB logger "src/computation/logger/logger.cpp")
file(GLOB tensor "src/computation/tensor/tensor.cpp" "src/computation/tensor/model_tensor.cpp")
#add_library(graph STATIC src/graph/graph.cpp)
add_subdirectory(contrib/googletest)
add_subdirectory(tests)
add_subdirectory(examples)

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
