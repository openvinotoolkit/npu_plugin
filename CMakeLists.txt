cmake_minimum_required(VERSION 3.10.0)

project(mcmComplier VERSION 1.0.0)

option(MCM_COMPILER_BUILD_PYTHON "Build Python wrappers" ON)
option(MCM_COMPILER_BUILD_TESTS "Build tests" ON)
option(MCM_COMPILER_BUILD_EXAMPLES "Build examples" ON)
option(MCM_COMPILER_BUILD_TOOLS "Build tools" ON)

if (NOT MCM_COMPILER_IS_SUBPROJECT)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-modules")

set(CMAKE_VERBOSE_MAKEFILE OFF) # for debugging

add_definitions(-DPROJECT_DIR=\"${PROJECT_SOURCE_DIR}\")

set(META_DIR_PATH "${PROJECT_BINARY_DIR}/meta")
include_directories(${META_DIR_PATH})


#############################################################################################
### Setup
#############################################################################################

if (NOT CMAKE_BUILD_TYPE)
    # Due to the current development pace, Debug is default build type
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_options(-std=c++11 --no-gnu-unique)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PROFILER FALSE CACHE BOOL "Enable integrated profiler")
set(LOG TRUE CACHE BOOL "Enable logger")
set(PROFILE_PHASE TRUE CACHE BOOL "Enable profiling at compilation phases level")
set(PROFILE_BASE TRUE CACHE BOOL "Enable profiling at base components level")
set(PROFILE_PASS TRUE CACHE BOOL "Enable profiling at compilation passes level")
set(PROFILE_COMP TRUE CACHE BOOL "Enable profiling at ops composition level")
set(PROFILE_ALGO TRUE CACHE BOOL "Enable profiling at internal algorithms level")
set(PROFILE_BULD TRUE CACHE BOOL "Enable profiling at building blocks level")
set(PROFILE_MATH TRUE CACHE BOOL "Enable profiling at tensor math operations level")

if (${LOG})
    add_definitions(-DMV_LOG_ENABLED)
endif()

if (${PROFILER})
    if (${UNIX})
        add_definitions(-DMV_PROFILER_ENABLED)
        include_directories(${PROJECT_SOURCE_DIR}/contrib/easy_profiler/easy_profiler_core/include)
        add_subdirectory(contrib/easy_profiler)

        if (${PROFILE_PHASE})
            add_definitions(-DMV_PROFILE_PHASE_ENABLED)
        endif()

        if (${PROFILE_BASE})
            add_definitions(-DMV_PROFILE_BASE_ENABLED)
        endif()

        if (${PROFILE_PASS})
            add_definitions(-DMV_PROFILE_PASS_ENABLED)
        endif()

        if (${PROFILE_COMP})
            add_definitions(-DMV_PROFILE_COMP_ENABLED)
        endif()

        if (${PROFILE_ALGO})
            add_definitions(-DMV_PROFILE_ALGO_ENABLED)
        endif()

        if (${PROFILE_BULD})
            add_definitions(-DMV_PROFILE_BULD_ENABLED)
        endif()

        if (${PROFILE_MATH})
            add_definitions(-DMV_PROFILE_MATH_ENABLED)
        endif()
    else()
        message("Ignoring PROFILER=ON flag, profiling only available for Linux/UNIX")
    endif()
endif()

#############################################################################################
### Test Coverage - off by default. Pass param CODE_COVERAGE=ON
### > cmake -DCODE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..
#############################################################################################

include(cmake-modules/code-coverage.cmake)
add_code_coverage()

#############################################################################################
### Find packages - uses Find*.cmake files in /cmake-modules dir
#############################################################################################

find_package(LEMON QUIET)
if(LEMON_FOUND)
    message(STATUS "Found LEMON installed on target system : ${LEMON_LIBRARY}")

    add_library(lemon UNKNOWN IMPORTED)
    set_target_properties(lemon PROPERTIES
        IMPORTED_LOCATION "${LEMON_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LEMON_INCLUDE_DIR}")
else()
    message(STATUS "LEMON library is not installed on target system, build it from the sources...")
    add_subdirectory(contrib/lemon)

    target_include_directories(lemon INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/contrib/lemon"
        "${CMAKE_CURRENT_BINARY_DIR}/contrib/lemon")
endif()

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(CMAKE_CROSSCOMPILING)
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(contrib/flatbuffers)

#############################################################################################
### Main build
#############################################################################################

if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>" "/GS")
else()
    add_compile_options("-Wall" "-Wextra" "-Wshadow" "-Wnon-virtual-dtor" "-pedantic" "$<$<CONFIG:RELEASE>:-O3>" "-fstack-protector-all")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/contrib/flatbuffers/include)

set(base
    "${PROJECT_SOURCE_DIR}/src/base/exception/argument_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/attribute_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/dtype_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/index_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/logged_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/logic_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/master_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/op_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/order_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/parsing_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/runtime_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/shape_error.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/exception/value_error.cpp"

    "${PROJECT_SOURCE_DIR}/src/base/json/array.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/bool.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/null.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/number_float.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/number_integer.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/object.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/string.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/value_content.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/value.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/json/json.cpp"

    "${PROJECT_SOURCE_DIR}/src/base/attribute_entry.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/binarizable.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/element.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/jsonable.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/printable.cpp"

    "${PROJECT_SOURCE_DIR}/src/computation/model/runtime_binary.cpp"

    "${PROJECT_SOURCE_DIR}/src/computation/op/op_entry.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/default.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/bin.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/float8.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/float16.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/float32.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/float64.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int2.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int2x.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int4.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int4x.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int8.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int16.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int32.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/int64.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/log.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/uint8.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/uint16.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/uint32.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_def/uint64.cpp"

    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_entry.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype_registry.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/dtype/dtype.cpp"

    # Single compilation unit for all Kmb attributes and operations in order
    "${PROJECT_SOURCE_DIR}/src/target/kmb/kmb_registry.cpp"

    "${PROJECT_SOURCE_DIR}/src/target/kmb/barrier_definition.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/barrier_deps.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/dma_direction.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/ppe_fixed_function.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/ppe_layer_type.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/ppe_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/workloads.cpp"

    "${PROJECT_SOURCE_DIR}/src/tensor/order/order_def/1D.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/order/order_def/2D.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/order/order_def/3D.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/order/order_def/4D.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/order/order_def/5D.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/order/order_registry.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/order/order.cpp"

    "${PROJECT_SOURCE_DIR}/src/tensor/data_element.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/math.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/quantization_params.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/shape.cpp"
    "${PROJECT_SOURCE_DIR}/src/tensor/tensor.cpp"

    "${PROJECT_SOURCE_DIR}/src/utils/parser/json_text.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils/serializer/file_buffer.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils/env_loader.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils/custom_math.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils/custom_strings.cpp"
)

set(model
    "${PROJECT_SOURCE_DIR}/src/computation/model/base_op_model.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/model/computation_model.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/model/control_model.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/model/data_model.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/model/group.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/model/model_element.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/flow/control_flow.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/flow/data_flow.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/op.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/resource/memory_allocator.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/resource/stage.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/lemon_graph_scheduler.cpp"
)

set(logger
    "${PROJECT_SOURCE_DIR}/src/logger/log_sender.cpp"
    "${PROJECT_SOURCE_DIR}/src/logger/logger.cpp"
)

set(pass
    "${PROJECT_SOURCE_DIR}/src/pass/pass_entry.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/pass_manager.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/pass_registry.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/pass_utils.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/initialization/global_config_params_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/compute_memory.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/initialization/assign_unique_op_id.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/create_source_structure.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/conv_dilation_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/fuse_passes.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/remove_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/replacement_passes.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/concat_as_implicit.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/reshape_as_implicit.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/update_implicit_layers_quatization_params.cpp"


    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_align_weights.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_control_flow.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/compute_tensors_quant.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/convert_to_task_graph.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/task_memory_location.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/strategy_from_layers_to_tensors.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_order_conversion.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/insert_barrier_tasks.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/dma_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/deallocation_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/assign_unique_task_id.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/splitting_across_clusters.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/store_layer_split_strategy.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/store_workload_strategy.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/implicit_operation_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/sparsity/sparsity_maps.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/sparsity/weights_tables.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/sparsity/eltwise_ops.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_align_to_16_channels.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/real_pad_tensors.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/streaming/spatial_split_streaming.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_align_slice_ops.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_quantize.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/force_serialize.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/finalization/max_topological_cut.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/allocate_memory_kmb.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/generate_workloads_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/tensor_graph_coloring.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/manipulate_barrier_refs.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/transitive_reduction.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/is_dag.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/schedule_helper.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/arrange_execution.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/optimization/graph_parameter_optimization.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/optimization/strategy_manager.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/optimization/MetaGraph.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/optimization/strategy_registry.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/optimization/kmb_default_configs.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/scheduling/scheduling_passes.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/serialization/generate_blob_kmb.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/serialization/generate_json.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/serialization/global_params_reset.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/validation/basic_checks.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/validation/generate_dot.cpp"


    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/lp_scheduler_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/remove_redundant_update_barriers.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/op_model_cutter.cpp"

    "${PROJECT_SOURCE_DIR}/src/target/target_descriptor.cpp"

    "${PROJECT_SOURCE_DIR}/src/graph/tensor_interference_graph.cpp"



)

set(compiler
    "${PROJECT_SOURCE_DIR}/src/compiler/compilation_descriptor"
    "${PROJECT_SOURCE_DIR}/src/compiler/compilation_unit"
    "${PROJECT_SOURCE_DIR}/src/target/kmb/runtime_model/runtime_model.cpp"
)

set(syscalls
   "${PROJECT_SOURCE_DIR}/src/pass/syscalls/convert_dot.cpp"
   "${PROJECT_SOURCE_DIR}/src/pass/syscalls/convert_flatbuffer.cpp"
)

if(CMAKE_CROSSCOMPILING)
    find_file(MCM_COMPILER_EXPORT_FILE mcmCompilerExecutables.cmake DOC "Path to the export file from a native build")

    if(MCM_COMPILER_EXPORT_FILE)
        message(STATUS "Cross-compiling, using executables from native build : ${MCM_COMPILER_EXPORT_FILE}")
        include(${MCM_COMPILER_EXPORT_FILE})
    endif()
endif()

add_subdirectory(meta)
add_subdirectory(schema)

if(NOT CMAKE_CROSSCOMPILING)
    # Export the generator targets to a file, so it can be imported (see above) by another build.
    export(TARGETS flatc gen_composition_api FILE ${CMAKE_BINARY_DIR}/mcmCompilerExecutables.cmake)
endif()

if (MSVC)
    add_library(cm OBJECT ${pass} ${compiler} $<TARGET_OBJECTS:metamodel>)
    add_library(model OBJECT ${model})

    add_dependencies(cm generate-comp-api)
    add_dependencies(cm generate-graphfile-headers)
    add_dependencies(model generate-comp-api)

    target_link_libraries(cm PUBLIC metamodel)
else()
    add_library(cm SHARED ${base} ${logger} ${pass} ${compiler} ${syscalls})
    add_library(model SHARED ${model})

    if(TARGET generate-comp-api)
        add_dependencies(cm generate-comp-api)
        add_dependencies(model generate-comp-api)
    endif()
    if(TARGET generate-graphfile-headers)
        add_dependencies(cm generate-graphfile-headers)
    endif()

    target_link_libraries(cm PUBLIC model metamodel flatbuffers)
    target_link_libraries(model PUBLIC lemon)

    if(NOT CMAKE_CROSSCOMPILING)
        if(MCM_COMPILER_BUILD_PYTHON)
            add_subdirectory(python/api)
        endif()
        if(MCM_COMPILER_BUILD_TESTS)
            add_subdirectory(contrib/googletest)
        endif()
    endif()
endif()

add_subdirectory(src/scheduler)
add_subdirectory(src/pass/lp_scheduler)
add_dependencies(cm lp_scheduler lp_scheduler_pass)
target_include_directories(cm
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/pass>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>)


target_include_directories(cm INTERFACE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/contrib/flatbuffers/include
    ${META_DIR_PATH})

if (${PROFILER} AND ${UNIX})
    target_link_libraries(cm easy_profiler)
endif()

if(NOT CMAKE_CROSSCOMPILING)
    # add_subdirectory(tools) # testing without
    if(MCM_COMPILER_BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()
    if(MCM_COMPILER_BUILD_TESTS)
        add_subdirectory(tests)
    endif()
    if(MCM_COMPILER_BUILD_TOOLS)
        add_subdirectory(tools)
    endif()
endif()
