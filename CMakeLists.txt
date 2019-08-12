cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-modules")
SET(CMAKE_VERBOSE_MAKEFILE OFF) # for debugging

#############################################################################################
### Setup
#############################################################################################
SET(PROJECT_VERSION_MAJOR 0)
SET(PROJECT_VERSION_MINOR 1)
SET(PROJECT_VERSION_PATCH 0)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
    endif()
endif()

add_compile_options(-std=c++11)

#############################################################################################
### Test Coverage - off by default. Pass param CODE_COVERAGE=ON
### > cmake -DCODE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..
#############################################################################################
include(cmake-modules/code-coverage.cmake)
add_code_coverage()


#############################################################################################
### Find packages - uses Find*.cmake files in /cmake-modules dir
#############################################################################################
find_package(METIS)
if(METIS_FOUND)
    set(METIS_LIBS ${METIS_LIBRARIES})
    include_directories(${METIS_INCLUDE_DIRS})
else()
    message(STATUS "Metis not found!")
endif(METIS_FOUND)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# find_package(Flatbuffers)
# if(FLATBUFFERS_FOUND)
#     include_directories(${FLATBUFFERS_INCLUDE_DIR})
# else()
#     message(STATUS "Flatbuffers not found!")
# endif(FLATBUFFERS_FOUND)

#############################################################################################
### Main build
#############################################################################################
if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
    SET(VALIDATION false)
else()
    add_compile_options("-Wall" "-Wextra" "-Wshadow" "-Wnon-virtual-dtor" "-pedantic" "$<$<CONFIG:RELEASE>:-O3>")
    SET(VALIDATION false CACHE BOOL "Flag to include additional external libaries for validation testing")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/contrib/flatbuffers/include)

if(VALIDATION)
    include_directories("/opt/movidius/caffe/include")
    include_directories("/opt/movidius/caffe/build/include")
    include_directories("/opt/movidius/caffe/build/include/caffe/proto")

    #Protobuf
    find_package(Protobuf REQUIRED)
    include_directories(${PROTOBUF_INCLUDE_DIR})

    #MVNC
    find_library(mvnc REQUIRED)

    #Boost
    find_package(Boost COMPONENTS regex system filesystem REQUIRED)

    #Caffe
    set(Caffe_INCLUDE_DIRS "/opt/movidius/caffe/build/include/caffe/proto;/opt/movidius/caffe/include;/opt/movidius/caffe/build/include")
    set(Caffe_LIBRARIES "/opt/movidius/caffe/build/lib/libcaffe.so")

    SET(pass_validation
        "${PROJECT_SOURCE_DIR}/src/pass/adaptation/generate_caffe.cpp"
    )
endif(VALIDATION)

SET(base
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/bool.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/double.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/dtype.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/implicit_flow.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/int.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/int64.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/mem_location.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/mv_element.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/order.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/shape.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_array_unsigned_short_2.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_array_unsigned_short_3.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_array_unsigned_short_4.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_map_std_string_element.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_set_std_string.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_size_t.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_string.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vec_mv_data_element.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vec_mv_element.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vector_double.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vector_int64.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vector_std_size_t.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vector_std_string.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/std_vector_unsigned.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/unsigned_short.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_def/unsigned.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/quantization_params.cpp"

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

    "${PROJECT_SOURCE_DIR}/src/base/attribute_entry.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/attribute_registry.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/binarizable.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/element.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/jsonable.cpp"
    "${PROJECT_SOURCE_DIR}/src/base/printable.cpp"

    "${PROJECT_SOURCE_DIR}/src/computation/model/runtime_binary.cpp"

    "${PROJECT_SOURCE_DIR}/src/computation/op/def/add.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/average_pool.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/batch_normalization.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/bias.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/clamp.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/concat.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/copy.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/constant.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/conv.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/conversion.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/depthwise_conv.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/divide.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/dropout.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/dummy.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/elu.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/fully_connected.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/identity.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/input.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/leaky_relu.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/local_response_normalization.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/matmul.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/max_pool.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/multiply.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/output.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/permute.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/prelu.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/region_yolo.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/relu.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/reorder.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/reorg_yolo.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/reshape.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/scale.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/slice.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/sigmoid.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/softmax.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/subtract.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/def/tanh.cpp"

    "${PROJECT_SOURCE_DIR}/src/computation/op/op_entry.cpp"
    "${PROJECT_SOURCE_DIR}/src/computation/op/op_registry.cpp"

    "${PROJECT_SOURCE_DIR}/src/target/keembay/barrier_definition.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/barrier_deps.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/dma_direction.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/ppe_fixed_function.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/ppe_layer_type.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/ppe_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/workloads.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/barrier_definition.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/barrier_deps.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/dma_direction.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/workloads.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/metisGraphStructure.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/ppe_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/attribute_def/ppe_fixed_function.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/tasks/def/barrier_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/tasks/def/deallocate_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/tasks/def/dma_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/tasks/def/dpu_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/tasks/def/upa_task.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/tasks/def/implicit_concat.cpp"


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

SET(model
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
    "${PROJECT_SOURCE_DIR}/src/target/myriadx/nce1.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/myriadx/nce1_utils.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/koala_graph_scheduler.cpp"
)

SET(logger
    "${PROJECT_SOURCE_DIR}/src/logger/log_sender.cpp"
    "${PROJECT_SOURCE_DIR}/src/logger/logger.cpp"
)

SET(pass
    "${PROJECT_SOURCE_DIR}/src/deployer/blob_serialization/blob_serializer.cpp"
    "${PROJECT_SOURCE_DIR}/src/deployer/blob_serialization/bRelocation.cpp"
    "${PROJECT_SOURCE_DIR}/src/deployer/blob_serialization/bTensor.cpp"
    "${PROJECT_SOURCE_DIR}/src/deployer/serializer.cpp"

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

    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/concat_as_implicit.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/update_implicit_layers_quatization_params.cpp"


    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/keembay_align_weights.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/keembay_control_flow.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/compute_tensors_quant.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/convert_to_task_graph.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/dpu_task_memory_location.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/keembay_order_conversion.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/insert_barrier_tasks.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/dma_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/deallocation_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/assign_unique_task_id.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/store_layer_split_strategy.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/strategy_from_layers_to_tensors.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/splitting_across_clusters.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/store_workload_strategy.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/implicit_operation_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/sparsity/sparsity_maps.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/sparsity/weights_tables.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/sparsity/eltwise_ops.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/keembay_align_to_16_channels.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/streaming/spatial_split_streaming.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/keembay_align_slice_ops.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/keembay_adaptation/force_serialize.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/max_topological_cut.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/align_data_order.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/allocate_memory_keem_bay.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/arrange_execution.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/format_mx_weights.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/fuse_relu_mx.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/generate_workloads_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/tensor_graph_coloring.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/manipulate_barrier_refs.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/mark_hardware_operations.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/myriad_x_paddings.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/optimize_convolutions.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/optimize_poolings.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/splits_over_h.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/tensor_field_storage.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/transitive_reduction.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/is_dag.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/finalization/schedule_helper.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/scheduling/scheduling_passes.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/serialization/generate_blob_keembay.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/serialization/generate_blob.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/serialization/generate_json.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/serialization/global_params_reset.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/validation/basic_checks.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/validation/generate_dot.cpp"

    "${PROJECT_SOURCE_DIR}/src/target/target_descriptor.cpp"

    "${PROJECT_SOURCE_DIR}/src/graph/tensor_interference_graph.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/koala/tinyxml/tinyxml.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/koala/tinyxml/tinyxmlerror.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/koala/tinyxml/tinyxmlparser.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/koala/tinyxml/tinystr.cpp"
)

 SET(compiler
    "${PROJECT_SOURCE_DIR}/src/compiler/compilation_descriptor"
    "${PROJECT_SOURCE_DIR}/src/compiler/compilation_unit"
    "${PROJECT_SOURCE_DIR}/src/utils/hardware_tests.cpp"
    "${PROJECT_SOURCE_DIR}/src/target/keembay/runtime_model/runtime_model.cpp"
)

SET(syscalls
   "${PROJECT_SOURCE_DIR}/src/pass/syscalls/convert_dot.cpp"
   "${PROJECT_SOURCE_DIR}/src/pass/syscalls/convert_flatbuffer.cpp"
)

IF(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling, assumes meta/ files already generated...")
    add_library(metamodel SHARED
        ${CMAKE_SOURCE_DIR}/meta/src/compositional_model.cpp
        ${CMAKE_SOURCE_DIR}/meta/src/op_model.cpp
        ${CMAKE_SOURCE_DIR}/meta/src/recorded_compositional_model.cpp
    )
    target_link_libraries(metamodel model)
ELSE()
    message(STATUS "Standard build...")
    add_subdirectory(meta)
ENDIF()

if (MSVC)
    add_library(cm OBJECT ${pass} ${compiler} $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel>)
        target_link_libraries(cm corebase metamodel)
    add_library(corebase OBJECT ${base} ${logger})
    add_library(model OBJECT ${model} $<TARGET_OBJECTS:corebase>)
    target_link_libraries(model corebase)
    add_dependencies(cm generate-comp-api)
    add_dependencies(cm generate-graphfile-headers)
    add_dependencies(model generate-comp-api)
else()
    if(VALIDATION)

        SET(deployer
            "${PROJECT_SOURCE_DIR}/src/utils/deployer/deployer_utils.cpp"
            "${PROJECT_SOURCE_DIR}/src/utils/deployer/executor.cpp"
        )
        add_library(cm SHARED ${base} ${logger} ${pass} ${compiler} ${pass_validation})
        add_library(corebase SHARED ${base} ${logger})
        add_library(model SHARED ${model})
        add_library(dep SHARED ${deployer})
        target_link_libraries(dep mvnc)
        target_link_libraries(cm corebase metamodel flatbuffers metis)
        add_dependencies(cm generate-comp-api)
        add_dependencies(model generate-comp-api)
        add_dependencies(cm generate-graphfile-headers)
        target_link_libraries(cm ${Caffe_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_REGEX_LIBRARY} ${PROTOBUF_LIBRARY})
        target_compile_definitions(cm PRIVATE CPU_ONLY=1)
        add_subdirectory(python/api)
        add_subdirectory(tests)
        add_subdirectory(contrib/googletest)
    else()
        add_library(cm SHARED ${base} ${logger} ${pass} ${compiler} ${syscalls})
        add_library(corebase SHARED ${base} ${logger})
        add_library(model SHARED ${model})


        IF(CMAKE_CROSSCOMPILING)
            #Flatbuffers and Metis not x-compiled yet
            target_link_libraries(cm corebase metamodel)
        ELSE()
            target_link_libraries(cm corebase metamodel flatbuffers metis)
            add_dependencies(cm generate-comp-api)
            add_dependencies(model generate-comp-api)
            add_dependencies(cm generate-graphfile-headers)
            add_subdirectory(python/api)
            add_subdirectory(tests)
            add_subdirectory(contrib/googletest)
        ENDIF(CMAKE_CROSSCOMPILING)

    endif(VALIDATION)
endif(MSVC)

IF(NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(examples)
    add_subdirectory(tools)
    add_subdirectory(schema)
    add_subdirectory(contrib/flatbuffers)
ENDIF()
