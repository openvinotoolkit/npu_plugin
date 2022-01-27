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

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "vpux_driver_compiler_adapter")
set(OBJ_TARGET_NAME "vpux_driver_compiler_adapter_obj")
set(STATIC_TARGET_NAME "vpux_driver_compiler_adapter_static")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

enable_warnings_as_errors(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/level-zero/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/level-zero-ext)

link_system_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_plugin_api
        ze_loader
        vpux_ngraph_transformations
        )

target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        vpux_al
        )


if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${OBJ_TARGET_NAME} FOR_TARGETS ${OBJ_TARGET_NAME} ALL)
    add_dependencies(${OBJ_TARGET_NAME} clang_format_${OBJ_TARGET_NAME})
endif()

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})
if(WIN32)
    ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
endif()

add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})

#
# targets install
#

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
