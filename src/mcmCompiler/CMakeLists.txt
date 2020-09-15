project(mcmComplier VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-modules")

#############################################################################################
### Setup
#############################################################################################

set(META_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/meta")

# Replace compiler flags
foreach(flag IN ITEMS "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
    string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()

# TODO: get rid of this define.
add_definitions(-DPROJECT_DIR=\"${PROJECT_SOURCE_DIR}\")
add_definitions(-DMV_LOG_ENABLED)

if(MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>" "/GS")
else()
    add_compile_options("-Wall" "-Wextra" "-Wshadow" "-Wnon-virtual-dtor" "-pedantic" "-fstack-protector-all")

    # TODO: check if we really need this.
    add_compile_options(--no-gnu-unique)
endif()

set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_CXX_VISIBILITY_PRESET default)

#############################################################################################
### Main build
#############################################################################################

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
    "${PROJECT_SOURCE_DIR}/src/utils/compression/hde.cpp"

    "${PROJECT_SOURCE_DIR}/contrib/huffman_encoding/src/Huffman.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/huffman_encoding/src/huffmanCodec.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/huffman_encoding/src/logging.cpp"
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

    "${PROJECT_SOURCE_DIR}/src/pass/quantization/cmconv_shapeBased_selection.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/quantization/propagate_fake_quantize.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/initialization/global_config_params_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/compute_memory.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/assign_unique_op_id.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/create_source_structure.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/assign_output_node.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/assign_input_node.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/initialization/use_reference_ops.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/conv_dilation_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/group_convolution_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/tile_operations.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/fuse_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/preprocess_PWL.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/remove_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/replacement_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/tile_operations.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/adaptation/ppe_replacement.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/concat_as_implicit.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/reshape_as_implicit.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/permute_as_implicit.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/update_implicit_layers_quatization_params.cpp"


    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_align_weights.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_control_flow.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/compute_output_dtype.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/place_operations.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/compute_tensors_quant.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/loadPwlQuantParams.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/convert_to_task_graph.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/bias_fixed_point_compute.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/task_memory_location.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/strategy_from_layers_to_tensors.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_order_conversion.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/insert_barrier_tasks.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/dma_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/deallocation_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/assign_unique_task_id.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/splitting_across_clusters.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/store_graph_optimizer_decisions.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/compute_sparsity_solution.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/store_workload_strategy.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/implicit_operation_passes.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/sparsity/sparsity_maps.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/sparsity/weights_tables.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/sparsity/eltwise_ops.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_align_to_16_channels.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/real_pad_tensors.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/streaming/spatial_split_streaming.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/kmb_quantize.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/force_serialize.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/mark_trailing_upatasks.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/kmb_adaptation/implicit_output_dtype_update.cpp"

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

    "${PROJECT_SOURCE_DIR}/src/pass/emulation/decide_precision.cpp"

    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/lp_scheduler_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/remove_redundant_update_barriers.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/op_model_cutter.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/barrier_scheduler_pass.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/force_spill_activation.cpp"

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

add_subdirectory(meta)

add_library(mcmCompiler SHARED
    ${base} ${model} ${logger} ${pass} ${compiler} ${syscalls}
    "${META_DIR_PATH}/src/compositional_model.cpp"
    "${META_DIR_PATH}/src/op_model.cpp"
    "${PROJECT_SOURCE_DIR}/src/pass/lp_scheduler/lp_scheduler_pass.cpp"
)

if(TARGET mcmCompilerGenCompAPI)
    add_dependencies(mcmCompiler mcmCompilerGenCompAPI)
endif()
if(TARGET mcmCompilerGenSchema)
    add_dependencies(mcmCompiler mcmCompilerGenSchema)
endif()

target_include_directories(mcmCompiler
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        "${PROJECT_SOURCE_DIR}/include"
        ${META_DIR_PATH}
        "${META_DIR_PATH}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/src/pass"
        "${PROJECT_SOURCE_DIR}/include/huffman_encoding"
)

target_link_libraries(mcmCompiler
    PUBLIC
        flatbuffers
        lemon
)
