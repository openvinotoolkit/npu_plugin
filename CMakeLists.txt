cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
    endif()
endif()

add_compile_options(-std=c++11)

if (MSVC)

    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
    SET(VALIDATION false)

else()

    add_compile_options("-Wall" "-Wextra" "-Wshadow" "-Wnon-virtual-dtor" "-fpermissive" "-pedantic" "$<$<CONFIG:RELEASE>:-O3>")
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

    file(GLOB pass_validation
        "src/pass/adaptation/generate_caffe.cpp"
    )

endif(VALIDATION)


file(GLOB base
    "src/base/attribute_registry.cpp"
    "src/base/attribute_entry.cpp"
    "src/base/attribute_def/*.cpp"
    "src/base/printable.cpp"
    "src/base/jsonable.cpp"
    "src/base/binarizable.cpp"
    "src/base/json/object.cpp"
    "src/base/json/array.cpp"
    "src/base/json/value.cpp"
    "src/base/json/value_content.cpp"
    "src/base/json/number_integer.cpp"
    "src/base/json/number_float.cpp"
    "src/base/json/string.cpp"
    "src/base/json/bool.cpp"
    "src/base/json/null.cpp"
    "src/target/keembay/barrier_definition.cpp"
    "src/target/keembay/barrier_deps.cpp"
    "src/target/keembay/dma_direction.cpp"
    "src/target/keembay/ppe_layer_type.cpp"
    "src/target/keembay/ppe_fixed_function.cpp"
    "src/target/keembay/workloads.cpp"
    "src/target/keembay/rectangle.cpp"
    "src/utils/parser/json_text.cpp"
    "src/utils/serializer/file_buffer.cpp"
    "src/deployer/Fp16Convert.cpp"
    "src/base/exception/*.cpp"
    "src/utils/env_loader.cpp"
    "src/utils/custom_math.cpp"
    "src/base/element.cpp"
    "src/tensor/shape.cpp"
    "src/tensor/data_element.cpp"
    "src/tensor/tensor.cpp"
    "src/tensor/binary_data.cpp"
    "src/tensor/quantization_params.cpp"
    "src/tensor/order/*.cpp"
    "src/tensor/order/order_def/*.cpp"
    "src/tensor/math.cpp"
    "src/tensor/dtype/*.cpp"
    "src/tensor/dtype/dtype_def/*.cpp"
    "src/computation/op/op_entry.cpp"
    "src/computation/op/op_registry.cpp"
    "src/computation/op/def/*.cpp"
    "src/computation/model/runtime_binary.cpp"
    "src/target/keembay/attribute_def/*.cpp"
    "src/target/keembay/tasks/def/*.cpp"
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
    "src/target/myriadx/nce1.cpp"
    "src/target/myriadx/nce1_utils.cpp"
    "src/computation/op/op.cpp"
    "src/computation/model/model_element.cpp"
)

file(GLOB logger
    "src/logger/logger.cpp"
    "src/logger/log_sender.cpp"
)

file(GLOB pass
    "src/deployer/*.cpp"
    "src/deployer/blob_serialization/*.cpp"
    "src/pass/pass_registry.cpp"
    "src/pass/pass_manager.cpp"
    "src/pass/pass_entry.cpp"
    "src/target/target_descriptor.cpp"
    "src/pass/common_functions.cpp"
    "src/pass/adaptation/fuse_passes.cpp"
    "src/pass/adaptation/remove_passes.cpp"
    "src/pass/adaptation/replacement_passes.cpp"
    "src/pass/adaptation/conv_dilation_pass.cpp"
    "src/pass/adaptation/create_source_structure.cpp"
    "src/pass/adaptation/assign_unique_task_id.cpp"
    "src/pass/adaptation/insert_barrier_tasks.cpp"
    "src/pass/adaptation/generate_keembay_sparse_structures.cpp"
    "src/pass/adaptation/assign_unique_op_id.cpp"
    "src/pass/adaptation/convert_to_task_graph.cpp"
    "src/pass/adaptation/dma_passes.cpp"
    "src/pass/adaptation/keembay_control_flow.cpp"
    "src/pass/finalization/generate_workloads_pass.cpp"
    "src/pass/adaptation/dpu_conv_pass.cpp"
    "src/pass/finalization/*.cpp"
    "src/pass/validation/*.cpp"
    "src/pass/serialization/*.cpp"
    "src/deployer/*.cpp"
    "src/deployer/blob_serialization/*.cpp"
    "contrib/koala/tinyxml/tinyxml.cpp"
    "contrib/koala/tinyxml/tinyxmlerror.cpp" 
    "contrib/koala/tinyxml/tinyxmlparser.cpp" 
    "contrib/koala/tinyxml/tinystr.cpp"
)

file(GLOB compiler
    "src/compiler/*"
    "src/utils/hardware_tests.cpp"
    "src/target/keembay/runtime_model/*.cpp"
)

add_subdirectory(meta)

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

    file(GLOB deployer
        "src/utils/deployer/*"
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
    #add_subdirectory(recordings)
else()
    add_library(cm SHARED ${base} ${logger} ${pass} ${compiler})

    add_library(corebase SHARED ${base} ${logger})
    add_library(model SHARED ${model})
    target_link_libraries(cm corebase metamodel flatbuffers metis)
    add_dependencies(cm generate-comp-api)
    add_dependencies(model generate-comp-api)
    add_dependencies(cm generate-graphfile-headers)
    add_subdirectory(python/api)
    add_subdirectory(tests)
    add_subdirectory(contrib/googletest)
    #add_subdirectory(recordings)
endif(VALIDATION)

endif(MSVC)

add_subdirectory(examples)
#add_subdirectory(docs/internal/tutorials)
add_subdirectory(tools)
add_subdirectory(schema)
add_subdirectory(contrib/flatbuffers)
