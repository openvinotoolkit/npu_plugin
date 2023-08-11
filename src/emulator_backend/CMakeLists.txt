#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "vpux_emulator_backend")
set(OBJ_TARGET_NAME "vpux_emulator_backend_obj")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include")

link_system_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_plugin_api
        vpux_utils)
target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        vpux_al
        kmb_utils
        flatbuffers
        vpux_emulator)

add_library(${TARGET_NAME} MODULE)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
