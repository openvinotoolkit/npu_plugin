#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

find_package(ZLIB)

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

set(TARGET_NAME npu_profiling_utils)

list(APPEND SOURCES
                common.cpp
                ../plugin/profiling_json.cpp
                ../plugin/profiling_meta.cpp
                ../plugin/profiling_parser.cpp
                ../ov/profiling.cpp
)

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})
vpux_enable_clang_format(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/plugin/
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/generated>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/include/vpux/compiler/profiling/generated>
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>

)

link_system_libraries(${TARGET_NAME}
    PUBLIC
        flatbuffers
        npu_elf
        openvino::reference
        openvino::runtime
        openvino::runtime::dev
)

target_compile_definitions(${TARGET_NAME} PUBLIC
        $<TARGET_PROPERTY:openvino::itt,INTERFACE_COMPILE_DEFINITIONS>)


add_dependencies(${TARGET_NAME}
    npu_mlir_compiler_schema
    npu_mlir_compiler_profiling_schema
)

add_dependencies(${TARGET_NAME}
    LLVMSupport
)

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE npu_core_utils npu_llvm_utils)
