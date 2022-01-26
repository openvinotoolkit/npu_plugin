#
# Copyright 2020 Intel Corporation.
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

set(TARGET_NAME "vpux_mcm_frontend")
set(OBJ_TARGET_NAME "vpux_mcm_frontend_obj")
set(STATIC_TARGET_NAME "vpux_mcm_frontend_static")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

enable_warnings_as_errors(${OBJ_TARGET_NAME})
# FIXME: this flag is forced while building in a single tree with OV
# but code cannot be compiled with the flag
if(NOT MSVC)
    target_compile_options(${OBJ_TARGET_NAME} PRIVATE -Wno-error=suggest-override)
endif()

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
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/meta/include"
)

link_system_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_legacy
        IE::inference_engine_transformations
        IE::inference_engine_plugin_api
        IE::ngraph_reference
        IE::vpu_common_lib
        vpux_mcm_compiler
        flatbuffers
        vpux_ngraph_transformations
)
target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        vpux_al
        kmb_utils
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${OBJ_TARGET_NAME}
        FOR_TARGETS ${OBJ_TARGET_NAME}
        EXCLUDE_PATTERNS ".*precomp.hpp"
        ALL
    )
    add_dependencies(${OBJ_TARGET_NAME} clang_format_${OBJ_TARGET_NAME})
endif()

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})

#
# targets install
#
 
install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
