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

# FIXME ENABLE_LTO leads to segmentation fault after 'calling fini libfrontend_mcm.so'
# if(ENABLE_LTO)
#     ie_enable_lto()
# endif()

set(TARGET_NAME "frontend_mcm")
set(OBJ_TARGET_NAME "frontend_mcm_obj")
set(STATIC_TARGET_NAME "frontend_mcm_static")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})

enable_warnings_as_errors(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
)

if(MSVC)
    target_compile_options(${OBJ_TARGET_NAME}
        PRIVATE
            /wd4146     # unary minus operator applied to unsigned type, result still unsigned
            /wd4996     # code uses a function, class member, variable, or typedef that's marked deprecated
    )
endif()

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/src/mcmCompiler/meta/include"
)

link_system_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_transformations
        IE::pugixml
        IE::ngraph_reference
        IE::vpu_common_lib
        mcmCompiler
)
target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        vpux
        kmb_utils
)

add_custom_target(mcm_compiler_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/src/mcmCompiler/config"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mcm_config"
    COMMENT "[MCM Compiler] Copy config files"
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${OBJ_TARGET_NAME} FOR_TARGETS ${OBJ_TARGET_NAME} ALL)
    add_dependencies(${OBJ_TARGET_NAME} clang_format_${OBJ_TARGET_NAME})
endif()

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
