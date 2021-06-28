#
# Copyright Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

set(TARGET_NAME "vpux_compiler")
set(TBLGEN_TARGET_NAME "${TARGET_NAME}_tblgen")
set(SCHEMA_TARGET_NAME "${TARGET_NAME}_schema")
set(DOCS_TARGET_NAME "${TARGET_NAME}_docs")
set(OBJ_TARGET_NAME "${TARGET_NAME}_obj")
set(STATIC_TARGET_NAME "${TARGET_NAME}_static")

#
# TBLGEN target
#

set(gen_base_src_dir "tblgen/vpux/compiler")
set(gen_base_dst_include_dir "include/vpux/compiler")
set(gen_base_dst_docs_dir "docs/generated")

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
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/ops_interfaces.md"
    )
    list(APPEND generated_docs "${subdir}/ops_interfaces.md")
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
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/attr_interfaces.md"
    )
    list(APPEND generated_docs "${subdir}/attr_interfaces.md")
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
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/type_interfaces.md"
    )
    list(APPEND generated_docs "${subdir}/type_interfaces.md")
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
        OUTPUT "${gen_base_dst_docs_dir}/dialect/${dialect}.md"
    )
    list(APPEND generated_docs "dialect/${dialect}.md")
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
        OUTPUT "${gen_base_dst_docs_dir}/${subdir}/passes.md"
    )
    list(APPEND generated_docs "${subdir}/passes.md")
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
add_passes("core" "Core")

add_dialect("const")
add_attr_interfaces("dialect/const")
add_attributes("const")

add_dialect("IE")
add_enum_attrs("IE")
add_struct_attrs("IE")
add_passes("dialect/IE" "IE")
add_rewriters("dialect/IE")

add_dialect("IERT")
add_passes("dialect/IERT" "IERT")
add_rewriters("dialect/IERT")

add_dialect("VPUIP")
add_op_interfaces("dialect/VPUIP")
add_enum_attrs("VPUIP")
add_struct_attrs("VPUIP")
add_types("VPUIP")
add_passes("dialect/VPUIP" "VPUIP")

add_passes("conversion" "Conversion")
add_rewriters("conversion")

file(GLOB_RECURSE TBLGEN_SOURCES CONFIGURE_DEPENDS "*.td")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TBLGEN_SOURCES})

vpux_add_tblgen_target(${TBLGEN_TARGET_NAME} ${TBLGEN_SOURCES})
set_target_properties(${TBLGEN_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TBLGEN_TARGET_NAME} mlir-headers mlir-generic-headers)

#
# DOCS target
#

set(docs_cmd_list)
foreach(doc_file IN LISTS generated_docs)
    set(gen_doc_file "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_docs_dir}/${doc_file}")
    set(out_doc_file "${CMAKE_CURRENT_SOURCE_DIR}/${gen_base_dst_docs_dir}/${doc_file}")

    get_filename_component(out_doc_dir ${out_doc_file} DIRECTORY)
    file(MAKE_DIRECTORY ${out_doc_dir})

    list(APPEND docs_cmd_list COMMAND ${CMAKE_COMMAND} -E copy ${gen_doc_file} ${out_doc_file})
endforeach()

add_custom_target(${DOCS_TARGET_NAME} ALL
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
    SRC_DIR "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated/schema"
    ARGS "--gen-name-strings"
)
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

#
# OBJECT target
#

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
set_target_properties(${OBJ_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${OBJ_TARGET_NAME}
    ${TBLGEN_TARGET_NAME}
    ${SCHEMA_TARGET_NAME}
)

ie_faster_build(${OBJ_TARGET_NAME}
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
        MLIRAffineTransforms
        MLIRAsync
        MLIRIR
        MLIRMemRef
        MLIROptLib
        MLIRPass
        MLIRQuant
        MLIRSCFTransforms
        MLIRStandard
        MLIRStandardOpsTransforms
        MLIRTransforms
        MLIRTranslation
        flatbuffers
        IE::ngraph
        IE::inference_engine
        IE::inference_engine_transformations
        # TODO: get rid of this dependency
        IE::vpu_common_lib
)
target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        vpux_utils
        vpux_al
)
if(ENABLE_PLAIDML)
    target_link_libraries(${OBJ_TARGET_NAME}
        PUBLIC
            pmlc::conversion::pxa_to_affine
            pmlc::conversion::tile_to_pxa
            pmlc::dialect::affinex::transforms
            pmlc::dialect::layer::ir
            pmlc::dialect::layer::transforms
            pmlc::dialect::pxa::ir
            pmlc::dialect::pxa::transforms
            pmlc::dialect::stdx::ir
            pmlc::dialect::stdx::transforms
            pmlc::dialect::tile::ir
            pmlc::dialect::tile::transforms
            pmlc::transforms
    )
endif()

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
