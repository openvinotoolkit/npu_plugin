#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
# Build configuration
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FIXME: LTO usage breaks vpux-translate for VPUIP blob generation in some cases (E#37558)
# if(ENABLE_LTO)
#     ie_enable_lto()
# endif()

set(TARGET_NAME "npu_mlir_compiler")
set(SCHEMA_TARGET_NAME "${TARGET_NAME}_schema")
set(PROFILING_SCHEMA_TARGET_NAME "${TARGET_NAME}_profiling_schema")
set(OBJ_TARGET_NAME "${TARGET_NAME}_obj")
set(STATIC_TARGET_NAME "${TARGET_NAME}_static")
set(MLIR_DEPS_NAME "npu_mlir_dependencies")
set(gen_base_dst_include_dir "include/vpux/compiler")

#
# SCHEMA target
#

vpux_add_flatc_target(${SCHEMA_TARGET_NAME}
    SRC_DIR "${VPUX_ELF_SOURCE_DIR}/src/schema"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated/schema"
    ARGS "--gen-name-strings")
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

#
# ELF profiling schema target
#

vpux_add_flatc_target(${PROFILING_SCHEMA_TARGET_NAME}
    SRC_DIR "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_utils/data"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/profiling/generated/schema"
    ARGS "--gen-object-api" "--reflect-names")
set_target_properties(${PROFILING_SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

if(NOT ENABLE_MLIR_COMPILER)
    return()
endif()

#
# Embed VPUNN models
#

find_package(Git REQUIRED)

execute_process(
    COMMAND ${GIT_EXECUTABLE} lfs pull
    WORKING_DIRECTORY "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel")

vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_0.vpunn"
    HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_0.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_0")
vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7.vpunn"
    HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7")
vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7.fast.vpunn"
    HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7_fast.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7_FAST")

#
# OBJECT target
#

file(GLOB SOURCES CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.hpp"
        "include/vpux/compiler/*.hpp")

# When vpux-plugin is built independently, MLIRConfig.cmake is under this dir
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_BINARY_DIR}/lib/cmake/mlir)
# When vpux-plugin is added as IE_EXTRA_MODULES, MLIRConfig.cmake is under this dir
list(APPEND CMAKE_PREFIX_PATH ${OpenVINO_BINARY_DIR}/lib/cmake/mlir)

find_package(MLIR REQUIRED CONFIG)
# Include .td files
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/thirdparty/llvm-project/mlir/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src/vpux_compiler/tblgen)
# Include header files under VPU37XX/api to avoid compiler warnings
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/dialect/VPU37XX/api)
include(cmake/vpux_tblgen.cmake)

# Reset MLIR_BINARY_DIR to redirect the generated docs in add_mlir_doc
set(MLIR_BINARY_DIR ${PROJECT_SOURCE_DIR}/src/vpux_compiler)
add_subdirectory(tblgen)

if(BUILD_COMPILER_FOR_DRIVER)
    addVersionDefines(${COMPILER_FOR_DRIVER_VERSION_FILE} CI_BUILD_NUMBER)
endif()

if(ENABLE_BITCOMPACTOR)
    set(BITCOMPACTOR_TARGET_OBJ $<TARGET_OBJECTS:bitcompactor>)
    set(BITCOMPACTOR_DEP bitcompactor)
    set(BITCOMPACTOR_INCLUDES $<TARGET_PROPERTY:bitcompactor,INTERFACE_INCLUDE_DIRECTORIES>)
else()
    set(BITCOMPACTOR_TARGET_OBJ "")
    set(BITCOMPACTOR_DEP "")
    set(BITCOMPACTOR_INCLUDES "")
endif()

set_property(GLOBAL PROPERTY VPU_SRC_LIB_LIST)
function(add_src_target name)
    get_property(lib_list GLOBAL PROPERTY VPU_SRC_LIB_LIST)
    list(APPEND lib_list ${name})
    set_property(GLOBAL PROPERTY VPU_SRC_LIB_LIST "${lib_list}")
endfunction()

# Construct source files after BITCOMPACTOR configuration
add_subdirectory(src)
get_property(VPU_SRC_LIBS GLOBAL PROPERTY VPU_SRC_LIB_LIST)

add_library(mlir-dependencies INTERFACE)
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(mlir-dependencies
    INTERFACE
        ${conversion_libs}
        MLIRExecutionEngine
        MLIROptLib)
target_include_directories(mlir-dependencies
    SYSTEM INTERFACE
        ${MLIR_INCLUDE_DIRS}
        ${VPUNN_INCLUDE_DIRS})

if(BUILD_SHARED_LIBS)
    add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
else()
    set(OBJ_TARGET_NAME ${TARGET_NAME})
    add_library(${OBJ_TARGET_NAME} STATIC ${SOURCES} ${COMPILER_FOR_DRIVER_VERSION_FILE})
    add_library(${MLIR_DEPS_NAME} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler_version.cpp)
endif()

add_dependencies(${OBJ_TARGET_NAME}
    ${SCHEMA_TARGET_NAME}
    ${PROFILING_SCHEMA_TARGET_NAME}
    mlir-doc)

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PUBLIC "src/precomp.hpp")
if(BUILD_SHARED_LIBS)
    vpux_enable_clang_format(${OBJ_TARGET_NAME})
endif()
enable_warnings_as_errors(${OBJ_TARGET_NAME} WIN_STRICT)
replace_noerror(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${NPU_DEVICE_NAME}\""
        "LIBRARY_OUTPUT_DIRECTORY=\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}\"")
target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
target_include_directories(${OBJ_TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/tblgen>
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/sw_runtime_kernels>)
target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        npu_al
        vpux_elf)
link_system_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        flatbuffers
        openvino::runtime
        openvino::runtime::dev
        IE::inference_engine_plugin_api
        IE::ngraph_reference)
target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        ${VPU_SRC_LIBS})

#
# MODULE target
#

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})
    if(WIN32)
        ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
        ov_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "VPU MLIR Compiler")
    endif()
endif()

#
# STATIC target
#

if(BUILD_SHARED_LIBS)
    add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
    target_link_libraries(${STATIC_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${OBJ_TARGET_NAME}>)
else()
    bundle_static_library(${OBJ_TARGET_NAME} inferenceStatic)
    bundle_static_library(${MLIR_DEPS_NAME} mlir-dependencies)
    add_dependencies(${MLIR_DEPS_NAME} ${OBJ_TARGET_NAME})
    add_library(${STATIC_TARGET_NAME} ALIAS ${OBJ_TARGET_NAME})
endif()

#
# targets install
#

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})

if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ${MLIR_DEPS_NAME} EXPORT OpenVINOTargets
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()
