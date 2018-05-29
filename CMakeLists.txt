cmake_minimum_required(VERSION 3.1.0)
project(fathom VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB allocator "src/graph/base_allocator.cpp")
file(GLOB computation_model 
    "src/computation/model/model.cpp"
    "src/computation/model/op_model.cpp"
    "src/computation/model/data_model.cpp"
    "src/computation/model/control_model.cpp"
    "src/computation/model/element.cpp"
    "src/computation/op/computation_op.cpp"
    "src/computation/op/kernel_op.cpp"
    "src/computation/model/attribute.cpp"
    "src/computation/flow/flow.cpp")

file(GLOB logger "src/computation/logger/logger.cpp")
file(GLOB tensor "src/computation/tensor/tensor.cpp" "src/computation/tensor/model_tensor.cpp")
file(GLOB pass "src/pass/deploy/deploy_pass.cpp")
#add_library(graph STATIC src/graph/graph.cpp)
add_subdirectory(contrib/googletest)
add_subdirectory(tests)
add_subdirectory(examples)

add_library(cm SHARED ${allocator} ${computation_model} ${logger} ${tensor} ${pass})

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
