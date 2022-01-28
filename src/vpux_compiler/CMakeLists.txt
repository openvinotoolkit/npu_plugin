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

replace_noerror()

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "vpux_mlir_compiler")
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
add_passes("core" "Core")

add_dialect("const")
add_attr_interfaces("dialect/const")
add_attributes("const")

add_dialect("IE")
add_op_interfaces("dialect/IE")
add_enum_attrs("IE")
add_struct_attrs("IE")
add_passes("dialect/IE" "IE")
add_rewriters("dialect/IE")

add_dialect("VPU")
add_op_interfaces("dialect/VPU")
add_enum_attrs("VPU")
add_struct_attrs("VPU")
add_types("VPU")
add_passes("dialect/VPU" "VPU")

add_dialect("IERT")
add_op_interfaces("dialect/IERT")
add_struct_attrs("IERT")
add_passes("dialect/IERT" "IERT")
add_rewriters("dialect/IERT")

add_dialect("VPUIP")
add_op_interfaces("dialect/VPUIP")
add_enum_attrs("VPUIP")
add_passes("dialect/VPUIP" "VPUIP")

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

if(BUILD_COMPILER_FOR_DRIVER)
    addVersionDefines(${COMPILER_FOR_DRIVER_VERSION_FILE} CI_BUILD_NUMBER)
endif()

if(BUILD_SHARED_LIBS)
    add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
else()
    set(OBJ_TARGET_NAME ${TARGET_NAME})
    add_library(${OBJ_TARGET_NAME} STATIC ${SOURCES} ${COMPILER_FOR_DRIVER_VERSION_FILE})
endif()

# FIXME: E#30420
# on one hand, adding bitcompactor and huffman_codec as sources
# makes clang_format corrupt those object files
# on the other, without them we do not have symbols defined in static build configuration
if(NOT BUILD_SHARED_LIBS)
    target_sources(${OBJ_TARGET_NAME} PRIVATE $<TARGET_OBJECTS:bitcompactor> $<TARGET_OBJECTS:huffman_codec>)
endif()
if(BUILD_SHARED_LIBS)
    vpux_enable_clang_format(${OBJ_TARGET_NAME})
endif()
set_target_properties(${OBJ_TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${OBJ_TARGET_NAME}
    ${TBLGEN_TARGET_NAME}
    ${SCHEMA_TARGET_NAME}
    bitcompactor
    huffman_codec
)

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PUBLIC "src/precomp.hpp"
)

enable_warnings_as_errors(${OBJ_TARGET_NAME} WIN_STRICT)
# FIXME: this flag is forced while building in a single tree with OV
# but code cannot be compiled with the flag
if(NOT MSVC)
    target_compile_options(${OBJ_TARGET_NAME}
        PUBLIC
            -Wno-error=suggest-override
    )
endif()


target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${VPUX_DEVICE_NAME}\""
        "LIBRARY_OUTPUT_DIRECTORY=\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}\""
)

# either copy sources or binaries to output folder
add_custom_target(copy-act-shave-runtime ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/sw_runtime_kernels/kernels/prebuild/act_shave_bin ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/act-kernels
        DEPENDS ${OBJ_TARGET_NAME})

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        $<TARGET_PROPERTY:bitcompactor,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:huffman_codec,INTERFACE_INCLUDE_DIRECTORIES>
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

target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        vpux_utils
        vpux_al
        vpux_elf
)

add_library(mlir-dependencies INTERFACE)
target_link_libraries(mlir-dependencies INTERFACE
    MLIRAffineToStandard
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
)
target_include_directories(mlir-dependencies
    SYSTEM INTERFACE 
        ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(${OBJ_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:mlir-dependencies>)

#
# MODULE target
#

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}> $<TARGET_OBJECTS:bitcompactor> $<TARGET_OBJECTS:huffman_codec>)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})
    if(WIN32)
        ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
    endif()
endif()

#
# STATIC target
#

if(BUILD_SHARED_LIBS)
    add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}> $<TARGET_OBJECTS:bitcompactor> $<TARGET_OBJECTS:huffman_codec>)
    target_link_libraries(${STATIC_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${OBJ_TARGET_NAME}>)
else()
    bundle_static_library(${OBJ_TARGET_NAME} mlir-dependencies)
    add_library(${STATIC_TARGET_NAME} ALIAS ${OBJ_TARGET_NAME})
endif()

#
# targets install
#

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
