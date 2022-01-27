#
# Copyright 2020-2021 Intel Corporation.
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

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME vpux_level_zero_backend)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE ${SOURCES})
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
        # IE::inference_engine
        IE::inference_engine_plugin_api
        vpux_utils
        ze_loader
 )

target_link_libraries(${TARGET_NAME}
    PRIVATE
        vpux_al
        kmb_utils
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()

#
# targets install
#

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
