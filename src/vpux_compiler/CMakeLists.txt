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

set(TARGET_NAME "vpux_mlir_compiler")
set(TBLGEN_TARGET_NAME "${TARGET_NAME}_tblgen")
set(SCHEMA_TARGET_NAME "${TARGET_NAME}_schema")
set(PROFILING_SCHEMA_TARGET_NAME "${TARGET_NAME}_profiling_schema")
set(DOCS_TARGET_NAME "${TARGET_NAME}_docs")
set(OBJ_TARGET_NAME "${TARGET_NAME}_obj")
set(STATIC_TARGET_NAME "${TARGET_NAME}_static")
set(MLIR_DEPS_NAME "vpux_mlir_dependencies")

#
# TBLGEN target
#

set(gen_base_src_dir "tblgen/vpux/compiler")
set(gen_base_dst_include_dir "include/vpux/compiler")
set(gen_base_dst_docs_dir "docs/chapters/generated")

set(generated_docs)

macro(add_op_interfaces subdir)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-op-interface-decls"
        SOURCE "${gen_base_src_dir}/${subdir}/ops_interfaces.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/ops_interfaces.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-op-interface-defs"
        SOURCE "${gen_base_src_dir}/${subdir}/ops_interfaces.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/ops_interfaces.cpp.inc"
    )

    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-op-interface-docs"
        SOURCE "${gen_base_src_dir}/${subdir}/ops_interfaces.td"
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/_ops_interfaces.md"
    )
    list(APPEND generated_docs "${subdir}/_ops_interfaces.md")
endmacro()

macro(add_attr_interfaces subdir)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-attr-interface-decls"
        SOURCE "${gen_base_src_dir}/${subdir}/attr_interfaces.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/attr_interfaces.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-attr-interface-defs"
        SOURCE "${gen_base_src_dir}/${subdir}/attr_interfaces.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/attr_interfaces.cpp.inc"
    )

    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-attr-interface-docs"
        SOURCE "${gen_base_src_dir}/${subdir}/attr_interfaces.td"
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/_attr_interfaces.md"
    )
    list(APPEND generated_docs "${subdir}/_attr_interfaces.md")
endmacro()

macro(add_type_interfaces subdir)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-type-interface-decls"
        SOURCE "${gen_base_src_dir}/${subdir}/type_interfaces.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/type_interfaces.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-type-interface-defs"
        SOURCE "${gen_base_src_dir}/${subdir}/type_interfaces.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/type_interfaces.cpp.inc"
    )

    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-type-interface-docs"
        SOURCE "${gen_base_src_dir}/${subdir}/type_interfaces.td"
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/_type_interfaces.md"
    )
    list(APPEND generated_docs "${subdir}/_type_interfaces.md")
endmacro()

macro(add_enum_attrs dialect)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-enum-decls"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/attributes.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/attributes/enums.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-enum-defs"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/attributes.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/attributes/enums.cpp.inc"
    )
endmacro()

macro(add_struct_attrs dialect)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-struct-attr-decls"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/attributes.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/attributes/structs.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-struct-attr-defs"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/attributes.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/attributes/structs.cpp.inc"
    )
endmacro()

macro(add_attributes dialect)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-attrdef-decls"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/attributes.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/attributes.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-attrdef-defs"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/attributes.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/attributes.cpp.inc"
    )
endmacro()

macro(add_types dialect)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-typedef-decls"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/types.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/types.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-typedef-defs"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/types.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/types.cpp.inc"
    )
endmacro()

macro(add_exclusive_types dialect)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-typedef-decls"
        EXTRA_ARGS "--typedefs-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/types.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/types.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-typedef-defs"
        EXTRA_ARGS "--typedefs-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/types.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/types.cpp.inc"
    )
endmacro()

macro(add_dialect dialect)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-dialect-decls"
        EXTRA_ARGS "-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/dialect.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-dialect-defs"
        EXTRA_ARGS "-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/dialect.cpp.inc"
    )

    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-op-decls"
        EXTRA_ARGS "-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/ops.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-op-defs"
        EXTRA_ARGS "-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/ops.cpp.inc"
    )

    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-dialect-doc"
        EXTRA_ARGS "-dialect=${dialect}"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_docs_dir}/dialect/_${dialect}.md"
    )
    list(APPEND generated_docs "dialect/_${dialect}.md")
endmacro()

macro(add_passes subdir name)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-pass-decls"
        EXTRA_ARGS "-name=${name}"
        SOURCE "${gen_base_src_dir}/${subdir}/passes.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/passes.hpp.inc"
    )

    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-pass-doc"
        SOURCE "${gen_base_src_dir}/${subdir}/passes.td"
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/_passes.md"
    )
    list(APPEND generated_docs "${subdir}/_passes.md")
endmacro()

macro(add_rewriters subdir)
    file(GLOB rewriters_sources
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS
        "${gen_base_src_dir}/${subdir}/rewriters/*.td"
    )

    foreach(src_file IN LISTS rewriters_sources)
        get_filename_component(src_file_name ${src_file} NAME_WE)

        vpux_add_tblgen_command(
            TOOL "MLIR"
            MODE "-gen-rewriters"
            SOURCE ${src_file}
            OUTPUT "${gen_base_dst_include_dir}/${subdir}/rewriters/generated/${src_file_name}.hpp.inc"
        )

        unset(src_file_name)
    endforeach()

    unset(rewriters_sources)
endmacro()

add_op_interfaces("core")
add_type_interfaces("core")
add_passes("core" "Core")

add_dialect("const")
add_attr_interfaces("dialect/const")
add_attributes("const")
add_passes("dialect/const" "Const")

add_dialect("IE")
add_op_interfaces("dialect/IE")
add_enum_attrs("IE")
add_struct_attrs("IE")
add_attributes("IE")
add_passes("dialect/IE" "IE")
add_rewriters("dialect/IE")
add_passes("VPU30XX/dialect/IE" "IE")
add_passes("VPU37XX/dialect/IE" "IE")

add_dialect("VPU")
add_attr_interfaces("dialect/VPU")
add_op_interfaces("dialect/VPU")
add_type_interfaces("dialect/VPU")
add_enum_attrs("VPU")
add_struct_attrs("VPU")
add_types("VPU")
add_attributes("VPU")
add_passes("dialect/VPU" "VPU")

add_dialect("EMU")
add_op_interfaces("dialect/EMU")
add_passes("dialect/EMU" "EMU")

add_dialect("IERT")
add_types("IERT")
add_rewriters("dialect/IERT")

add_dialect("VPUIP")
add_op_interfaces("dialect/VPUIP")
add_attributes("VPUIP")
add_types("VPUIP")
add_enum_attrs("VPUIP")
add_struct_attrs("VPUIP")
add_passes("dialect/VPUIP" "VPUIP")
add_rewriters("dialect/VPUIP")

add_dialect("VPURT")
add_op_interfaces("dialect/VPURT")
add_types("VPURT")
add_enum_attrs("VPURT")
add_passes("dialect/VPURT" "VPURT")

add_dialect("ELF")
add_op_interfaces("dialect/ELF")
add_enum_attrs("ELF")
add_types("ELF")
add_passes("dialect/ELF" "ELF")

add_dialect("VPUMI37XX")
add_op_interfaces("dialect/VPUMI37XX")
add_attributes("VPUMI37XX")
add_enum_attrs("VPUMI37XX")
add_struct_attrs("VPUMI37XX")
add_types("VPUMI37XX")
add_passes("dialect/VPUMI37XX" "VPUMI37XX")

add_dialect("VPURegMapped")
add_op_interfaces("dialect/VPURegMapped")
add_attributes("VPURegMapped")
add_types("VPURegMapped")
add_enum_attrs("VPURegMapped")
add_passes("dialect/VPURegMapped" "VPURegMapped")

add_dialect("VPU37XX")
add_exclusive_types("VPU37XX")

add_passes("conversion" "Conversion")
add_rewriters("conversion")

file(GLOB_RECURSE TBLGEN_SOURCES CONFIGURE_DEPENDS "*.td")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TBLGEN_SOURCES})

vpux_add_tblgen_target(${TBLGEN_TARGET_NAME} ${TBLGEN_SOURCES})
set_target_properties(${TBLGEN_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TBLGEN_TARGET_NAME} mlir-headers mlir-generic-headers)

# Touch all *.td files to update their time and force them to be rebuild.
add_custom_target(rebuild_tblgen COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${TBLGEN_SOURCES})

#
# DOCS target
#

set(docs_cmd_list)
foreach(doc_file IN LISTS generated_docs)
    set(gen_doc_file "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_docs_dir}/${doc_file}")
    set(out_doc_file "${CMAKE_CURRENT_SOURCE_DIR}/${gen_base_dst_docs_dir}/${doc_file}")

    get_filename_component(out_doc_dir ${out_doc_file} DIRECTORY)

    list(APPEND docs_cmd_list COMMAND ${CMAKE_COMMAND} -E make_directory ${out_doc_dir})
    list(APPEND docs_cmd_list COMMAND ${CMAKE_COMMAND} -E copy ${gen_doc_file} ${out_doc_file})
endforeach()

add_custom_target(${DOCS_TARGET_NAME} ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/${gen_base_dst_docs_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/${gen_base_dst_docs_dir}"
    ${docs_cmd_list}
    COMMENT "[DOCS] ${TARGET_NAME}"
)

add_dependencies(${DOCS_TARGET_NAME}
    ${TBLGEN_TARGET_NAME}
)

#
# SCHEMA target
#

vpux_add_flatc_target(${SCHEMA_TARGET_NAME}
    SRC_DIR "${VPUX_ELF_SOURCE_DIR}/src/schema"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated/schema"
    ARGS "--gen-name-strings"
)
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

#
# ELF profiling schema target
#

vpux_add_flatc_target(${PROFILING_SCHEMA_TARGET_NAME}
    SRC_DIR "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_utils/data"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/profiling/generated/schema"
    ARGS "--gen-object-api"
)
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
    WORKING_DIRECTORY "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel"
)

vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_0.vpunn"
    HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_0.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_0"
)
vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7.vpunn"
    HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7"
)
vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7.fast.vpunn"
    HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7_fast.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7_FAST"
)

#
# OBJECT target
#

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

if(BUILD_COMPILER_FOR_DRIVER)
    addVersionDefines(${COMPILER_FOR_DRIVER_VERSION_FILE} CI_BUILD_NUMBER)
endif()

if(BUILD_SHARED_LIBS)
    add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
else()
    set(OBJ_TARGET_NAME ${TARGET_NAME})
    add_library(${OBJ_TARGET_NAME} STATIC ${SOURCES} ${COMPILER_FOR_DRIVER_VERSION_FILE})
    add_library(${MLIR_DEPS_NAME} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler_version.cpp)
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

if(NOT BUILD_SHARED_LIBS)
    target_sources(${OBJ_TARGET_NAME} PRIVATE
        ${BITCOMPACTOR_TARGET_OBJ}
        $<TARGET_OBJECTS:huffman_codec>
        $<TARGET_OBJECTS:act_shave_kernels_lib>
    )
endif()
if(BUILD_SHARED_LIBS)
    vpux_enable_clang_format(${OBJ_TARGET_NAME})
endif()
set_target_properties(${OBJ_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${OBJ_TARGET_NAME}
    ${TBLGEN_TARGET_NAME}
    ${SCHEMA_TARGET_NAME}
    ${PROFILING_SCHEMA_TARGET_NAME}
    ${BITCOMPACTOR_DEP}
    huffman_codec
)

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PUBLIC "src/precomp.hpp"
)

enable_warnings_as_errors(${OBJ_TARGET_NAME} WIN_STRICT)
replace_noerror(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${VPUX_DEVICE_NAME}\""
        "LIBRARY_OUTPUT_DIRECTORY=\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}\""
)

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:inferenceStatic,INCLUDE_DIRECTORIES>>
    PRIVATE
        ${BITCOMPACTOR_INCLUDES}
        $<TARGET_PROPERTY:huffman_codec,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:IE::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
)

target_include_directories(${OBJ_TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/sw_runtime_kernels>
)



link_system_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        flatbuffers
        openvino::runtime
        openvino::runtime::dev
        IE::inference_engine_plugin_api
        IE::ngraph_reference
        vpux_ngraph_transformations
        vpux_utils
)

target_link_libraries(${OBJ_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:inferenceStatic>)

target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        vpux_utils
        vpux_al
        vpux_elf
)

add_library(mlir-dependencies INTERFACE)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(mlir-dependencies INTERFACE
    ${conversion_libs}
    MLIRExecutionEngine
    MLIROptLib
)
target_include_directories(mlir-dependencies
    SYSTEM INTERFACE
        ${MLIR_INCLUDE_DIRS}
        ${VPUNN_INCLUDE_DIRS}
)

target_link_libraries(${OBJ_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:mlir-dependencies>)

#
# MODULE target
#

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE
        ${BITCOMPACTOR_TARGET_OBJ}
        $<TARGET_OBJECTS:huffman_codec>
        $<TARGET_OBJECTS:act_shave_kernels_lib>
    )
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
    add_library(${STATIC_TARGET_NAME} STATIC
        $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>
        ${BITCOMPACTOR_TARGET_OBJ}
        $<TARGET_OBJECTS:huffman_codec>
        $<TARGET_OBJECTS:act_shave_kernels_lib>
    )
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
