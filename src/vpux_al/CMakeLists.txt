#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME npu_al)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_warnings_as_errors(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include>
)

set(VPUX_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp)
addVersionDefines(${VPUX_VERSION_FILE} CI_BUILD_NUMBER)
# FIXME: begin
# E#-34790
# backend shouldn't refer to plugin headers
# temporary work-around until everything related to the backend moved
# from the plugin to the backend
target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_plugin/include>
)

# FIXME: end

link_system_libraries(${TARGET_NAME}
    PRIVATE
        npu_ov_utils
        npu_llvm_utils
)

if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            openvino::runtime
    )
else()
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            openvino::runtime::dev
            openvino::runtime
    )

    if(ENABLE_MLIR_COMPILER)
        target_link_libraries(${TARGET_NAME}
            PUBLIC
                npu_mlir_compiler
                npu_mlir_dependencies
        )
    endif()

    if(ENABLE_DRIVER_COMPILER_ADAPTER)
        target_link_libraries(${TARGET_NAME}
            PUBLIC
                npu_driver_compiler_adapter
        )
    endif()

    if(ENABLE_ZEROAPI_BACKEND)
        target_link_libraries(${TARGET_NAME}
            PUBLIC
                npu_level_zero_backend
        )

        target_include_directories(${TARGET_NAME}
            PUBLIC
                $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/zero_backend/include>
        )
    endif()

    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif ()

target_link_libraries(${TARGET_NAME}
    PRIVATE
        npu_ov_utils
        npu_llvm_utils
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()

#
# targets install
#

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()
