cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(default_build_type "Release")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(MCM_HOME "$ENV{MCM_HOME}")
if(NOT MCM_HOME)
    message(STATUS "MCM_HOME at ${MCM_HOME}")
    message(FATAL_ERROR "Undefined MCM_HOME env variable")
endif()

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

file(GLOB base
    "src/base/attribute.cpp"
    "src/base/printable.cpp"
    "src/base/jsonable.cpp"
    "src/base/stream/ostream.cpp"
    "src/base/json/object.cpp"
    "src/base/json/array.cpp"
    "src/base/json/value.cpp"
    "src/base/json/value_content.cpp"
    "src/base/json/number_integer.cpp"
    "src/base/json/number_float.cpp"
    "src/base/json/string.cpp"
    "src/base/json/bool.cpp"
    "src/base/json/null.cpp"
    "src/utils/parser/json_text.cpp"
    "src/base/exception/*.cpp"
    "src/utils/*.cpp"
    "src/base/singleton.cpp"
    "src/base/order/*.cpp"
)

file(GLOB tensor
    "src/computation/tensor/shape.cpp"
    "src/computation/tensor/tensor.cpp"
    "src/computation/tensor/math.cpp"
)

file(GLOB computation_model
    "src/computation/model/computation_model.cpp"
    "src/api/compositional_model.cpp"
    "src/computation/model/op_model.cpp"
    "src/computation/model/data_model.cpp"
    "src/computation/model/control_model.cpp"
    "src/computation/model/computation_element.cpp"
    "src/computation/op/computation_op.cpp"
    "src/computation/op/kernel_op.cpp"
    "src/computation/model/computation_group.cpp"
    "src/computation/flow/flow.cpp"
    "src/computation/flow/control_flow.cpp"
    "src/computation/flow/data_flow.cpp"
    "src/computation/resource/memory_allocator.cpp"
    "src/computation/resource/computation_stage.cpp"
    "src/computation/op/source_op.cpp"
    "src/computation/op/sink_op.cpp"
    "src/computation/op/pool2d_op.cpp"
    "src/computation/op/activation_op.cpp"
    "src/computation/op/eltwise_op.cpp"
    "src/deployer/*.cpp"
    "src/deployer/blob_serialization/*.cpp"
)

file(GLOB ops
    "src/computation/op/def/add.cpp"
    "src/computation/op/def/avgpool2d.cpp"
    "src/computation/op/def/batch_norm.cpp"
    "src/computation/op/def/bias.cpp"
    "src/computation/op/def/concat.cpp"
    "src/computation/op/def/constant.cpp"
    "src/computation/op/def/conv2d.cpp"
    "src/computation/op/def/divide.cpp"
    "src/computation/op/def/matmul.cpp"
    "src/computation/op/def/input.cpp"
    "src/computation/op/def/maxpool2d.cpp"
    "src/computation/op/def/multiply.cpp"
    "src/computation/op/def/output.cpp"
    "src/computation/op/def/relu.cpp"
    "src/computation/op/def/reshape.cpp"
    "src/computation/op/def/scale.cpp"
    "src/computation/op/def/softmax.cpp"
    "src/computation/op/def/subtract.cpp"
    "src/computation/op/def/fully_connected.cpp"
    "src/computation/op/def/conversion.cpp"
)

file(GLOB logger
    "src/logger/logger.cpp"
    "src/logger/stdout_logger.cpp"
)

file(GLOB pass
    "src/pass/pass_registry.cpp"
    "src/pass/pass_manager.cpp"
    "src/target/target_descriptor.cpp"
    "src/pass/adaptation/*.cpp"
    "src/pass/finalization/*.cpp"
    "src/pass/validation/*.cpp"
    "src/pass/serialization/*.cpp"
    "src/deployer/*.cpp"
    "src/deployer/blob_serialization/*.cpp"
)

file(GLOB compiler
    "src/compiler/*"
)

add_subdirectory(contrib/googletest)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(docs/internal/tutorials)
add_subdirectory(python/api)

add_library(cm SHARED ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${base} ${tensor} ${compiler})