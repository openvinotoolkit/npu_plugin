#
# Copyright 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

set(TARGET_NAME "vpux_compiler")
set(TBLGEN_TARGET_NAME "${TARGET_NAME}_tblgen")
set(SCHEMA_TARGET_NAME "${TARGET_NAME}_schema")
set(OBJ_TARGET_NAME "${TARGET_NAME}_obj")
set(STATIC_TARGET_NAME "${TARGET_NAME}_static")

#
# TBLGEN target
#

set(gen_base_src_dir "tblgen/vpux/compiler")
set(gen_base_dst_include_dir "include/vpux/compiler")
set(gen_base_dst_docs_dir "docs/generated")

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

    if (UNIX)
        vpux_add_tblgen_command(
            TOOL "MLIR"
            MODE "-gen-op-interface-docs"
            SOURCE "${gen_base_src_dir}/${subdir}/ops_interfaces.td"
            OUTPUT "${gen_base_dst_docs_dir}/${subdir}/ops_interfaces.md"
        )
    endif()
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
        MODE "-gen-op-decls"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/ops.hpp.inc"
    )
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-op-defs"
        SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
        OUTPUT "${gen_base_dst_include_dir}/dialect/${dialect}/generated/ops.cpp.inc"
    )

    if (UNIX)
        vpux_add_tblgen_command(
            TOOL "MLIR"
            MODE "-gen-dialect-doc"
            SOURCE "${gen_base_src_dir}/dialect/${dialect}/ops.td"
            OUTPUT "${gen_base_dst_docs_dir}/dialect/${dialect}.md"
        )
    endif()
endmacro()

macro(add_passes subdir name)
    vpux_add_tblgen_command(
        TOOL "MLIR"
        MODE "-gen-pass-decls"
        EXTRA_ARGS "-name=${name}"
        SOURCE "${gen_base_src_dir}/${subdir}/passes.td"
        OUTPUT "${gen_base_dst_include_dir}/${subdir}/generated/passes.hpp.inc"
    )

    if (UNIX)
        vpux_add_tblgen_command(
            TOOL "MLIR"
            MODE "-gen-pass-doc"
            SOURCE "${gen_base_src_dir}/${subdir}/passes.td"
            OUTPUT "${gen_base_dst_docs_dir}/${subdir}/passes.md"
        )
    endif()
endmacro()

macro(add_rewriters subdir)
    file(GLOB rewriters_sources
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
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

add_dialect("IE")
add_enum_attrs("IE")
add_passes("dialect/IE" "IE")
add_rewriters("dialect/IE")

add_dialect("IERT")
add_passes("dialect/IERT" "IERT")

add_dialect("VPUIP")
add_op_interfaces("dialect/VPUIP")
add_enum_attrs("VPUIP")
add_struct_attrs("VPUIP")
add_types("VPUIP")
add_passes("dialect/VPUIP" "VPUIP")

add_passes("conversion" "Conversion")
add_rewriters("conversion")

add_passes("pipelines" "Pipelines")

file(GLOB_RECURSE TBLGEN_SOURCES "*.td")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TBLGEN_SOURCES})

vpux_add_tblgen_target(${TBLGEN_TARGET_NAME} ${TBLGEN_SOURCES})
set_target_properties(${TBLGEN_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TBLGEN_TARGET_NAME} mlir-headers mlir-generic-headers)

#
# SCHEMA target
#

vpux_add_flatc_target(${SCHEMA_TARGET_NAME}
    SRC_DIR "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated/schema"
    ARGS "--gen-name-strings"
)
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

#
# OBJECT target
#

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
set_target_properties(${OBJ_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${OBJ_TARGET_NAME}
    ${TBLGEN_TARGET_NAME}
    ${SCHEMA_TARGET_NAME}
)

ie_faster_build(${OBJ_TARGET_NAME}
    UNITY
    PCH PUBLIC "src/precomp.hpp"
)

enable_warnings_as_errors(${OBJ_TARGET_NAME} WIN_STRICT)
vpux_enable_clang_format(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
)

target_include_directories(${OBJ_TARGET_NAME}
    SYSTEM PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        ${MLIR_INCLUDE_DIRS}
)
target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

link_system_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        MLIRIR
        MLIROptLib
        MLIRPass
        MLIRQuant
        MLIRStandard
        MLIRTransforms
        MLIRTranslation
        MLIRStandardOpsTransforms
        MLIRLinalg
        MLIRShape
        flatbuffers
        # TODO: get rid of this dependency
        IE::vpu_common_lib
)
target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        vpux_utils
        vpux_al
)

#
# MODULE target
#

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

#
# STATIC target
#

add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
