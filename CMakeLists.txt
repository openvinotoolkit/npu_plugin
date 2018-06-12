cmake_minimum_required(VERSION 3.5.0)
project(fathom VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB allocator "src/graph/base_allocator.cpp")
file(GLOB computation_model 
    "src/computation/model/model.cpp"
    "src/computation/api/compositional_model.cpp"
    "src/computation/model/op_model.cpp"
    "src/computation/model/data_model.cpp"
    "src/computation/model/control_model.cpp"
    "src/computation/model/element.cpp"
    "src/computation/op/computation_op.cpp"
    "src/computation/op/kernel_op.cpp"
    "src/computation/model/attribute.cpp"
    "src/computation/flow/flow.cpp"
    "src/computation/op/source_op.cpp"
    "src/computation/op/sink_op.cpp"
    "src/computation/op/pool2d_op.cpp"
    "src/computation/op/activation_op.cpp"
    "src/computation/op/eltwise_op.cpp"
    "src/deployer/ostream.cpp")

file(GLOB logger "src/computation/logger/logger.cpp")
file(GLOB pass "src/pass/deploy/deploy_pass.cpp")
#add_library(graph STATIC src/graph/graph.cpp)
add_subdirectory(contrib/googletest)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(python/api)

add_library(cm SHARED ${allocator} ${computation_model} ${logger} ${pass})

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
