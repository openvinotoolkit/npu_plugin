cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

add_compile_options(-std=c++11)

if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options("-Wall" "-Wextra" "-Wshadow" "-Wnon-virtual-dtor" "-pedantic" "$<$<CONFIG:RELEASE>:-O3>")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB base
    "src/base/attribute_registry.cpp"
    "src/base/attribute_entry.cpp"
    "src/base/attribute_def/*.cpp"
    "src/base/printable.cpp"
    "src/base/jsonable.cpp"
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
    "src/utils/env_loader.cpp"
    "src/utils/custom_math.cpp"
    "src/base/element.cpp"
    "src/tensor/shape.cpp"
    "src/tensor/tensor.cpp"
    "src/tensor/math.cpp"
    "src/tensor/order.cpp"
    "src/tensor/order/column_major_planar.cpp"
    "src/tensor/order/column_major.cpp"
    "src/tensor/order/row_major_planar.cpp"
    "src/tensor/order/row_major.cpp"
    "src/tensor/order/row_interleaved.cpp"
    "src/tensor/dtype.cpp"
    "src/computation/op/op_entry.cpp"
    "src/computation/op/op_registry.cpp"
    "src/computation/op/def2/*.cpp"
)

file (GLOB model
    "src/computation/model/computation_model.cpp"
    "src/computation/model/base_op_model.cpp"
    "src/computation/model/op_model.cpp"
    "src/computation/model/data_model.cpp"
    "src/computation/model/control_model.cpp"
    "src/computation/model/computation_element.cpp"
    "src/api/compositional_model.cpp"
    "src/utils/recorded_compositional_model.cpp"
    "src/computation/model/group.cpp"
    "src/computation/flow/control_flow.cpp"
    "src/computation/flow/data_flow.cpp"
    "src/computation/resource/memory_allocator.cpp"
    "src/computation/resource/stage.cpp"
    "src/computation/resource/nce1.cpp"
    "src/computation/resource/nce1_utils.cpp"
    "src/computation/op/op.cpp"
    "src/computation/model/model_element.cpp"
)

file(GLOB logger
    "src/logger/logger.cpp"
    "src/logger/log_sender.cpp"
)

file(GLOB pass
    "src/deployer/*.cpp"
    #"src/deployer/blob_serialization/*.cpp"
    "src/pass/pass_registry.cpp"
    "src/pass/pass_manager.cpp"
    "src/pass/pass_entry.cpp"
    "src/target/target_descriptor.cpp"
    "src/pass/adaptation/*.cpp"
    "src/pass/finalization/*.cpp"
    "src/pass/validation/*.cpp"
    #[["src/pass/serialization/*.cpp"
    "src/deployer/*.cpp"
    "src/deployer/blob_serialization/*.cpp"]]
)

file(GLOB compiler
    "src/compiler/*"
)

add_subdirectory(meta)

if (MSVC)
    add_library(cm OBJECT ${model})
    add_library(corebase OBJECT ${base} ${logger})
    add_library(model OBJECT ${model})
    target_link_libraries(cm corebase metamodel)
    add_dependencies(cm generate-comp-api)
    add_dependencies(model generate-comp-api)
else()
    add_library(cm SHARED ${model})
    add_library(corebase SHARED ${base} ${logger})
    add_library(model SHARED ${model})
    target_link_libraries(cm corebase metamodel)
    add_dependencies(cm generate-comp-api)
    add_dependencies(model generate-comp-api)
    #add_subdirectory(python/api)
	add_subdirectory(tests)
	add_subdirectory(contrib/googletest)
endif()

add_subdirectory(examples)
#add_subdirectory(docs/internal/tutorials)

