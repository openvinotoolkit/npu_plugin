#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME npu_level_zero_backend)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE ${SOURCES})
    if(WIN32)
        ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
        ov_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "VPU Plugin L0 backend")
    endif()
else()
    add_library(${TARGET_NAME} STATIC ${SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

enable_warnings_as_errors(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/thirdparty/level-zero/include
        ${PROJECT_SOURCE_DIR}/thirdparty/level-zero-ext)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        openvino::runtime::dev
        ze_loader
 )

target_link_libraries(${TARGET_NAME}
    PRIVATE
        npu_al
        npu_ov_utils
        npu_llvm_utils
        npu_plugin_utils
        npu_profiling_utils
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()

#
# targets install
#

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT})
