#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

find_package(ZLIB)

set(TARGET_NAME npu_ov_utils)

list(APPEND SOURCES
        config.cpp
        loop.cpp
        logger_adapter.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})
vpux_enable_clang_format(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/IE/
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/LLVM/
)

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(${TARGET_NAME} PUBLIC
        $<TARGET_PROPERTY:openvino::itt,INTERFACE_COMPILE_DEFINITIONS>)

link_system_libraries(${TARGET_NAME}
    PUBLIC
        openvino::reference
        openvino::runtime
        openvino::runtime::dev
)

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE npu_core_utils npu_llvm_utils)
