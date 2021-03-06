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

#if(ENABLE_LTO)
#    ie_enable_lto()
#endif()

set(TARGET_NAME "emulator_backend")
set(OBJ_TARGET_NAME "emulator_backend_obj")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
# FIXME: enable once MCMCompiler respects warning as errors
# enable_warnings_as_errors(${OBJ_TARGET_NAME})

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
        IE::vpu_common_lib)
target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        vpux_al
        kmb_utils
        flatbuffers
        mcmCompiler)

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
