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

set(TARGET_NAME "vpux_vpual_backend")
set(OBJ_TARGET_NAME "vpux_vpual_backend_obj")
set(STATIC_TARGET_NAME "vpux_vpual_backend_static")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})

enable_warnings_as_errors(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2")

link_system_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        ${VPUAL_NN_LIBRARY}
        ${VPUMGR_LIBRARY}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_plugin_api
        vpux_utils)
target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        vpux_al
        kmb_utils
        vpux_mcm_compiler)

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

if (ENABLE_TESTS)
    add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
    target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${OBJ_TARGET_NAME} FOR_TARGETS ${OBJ_TARGET_NAME} ALL)
    add_dependencies(${OBJ_TARGET_NAME} clang_format_${OBJ_TARGET_NAME})
endif()

#
# targets install
#

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
