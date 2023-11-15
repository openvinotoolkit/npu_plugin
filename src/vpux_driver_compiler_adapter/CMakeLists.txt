#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "npu_driver_compiler_adapter")
set(OBJ_TARGET_NAME "npu_driver_compiler_adapter_obj")
set(STATIC_TARGET_NAME "npu_driver_compiler_adapter_static")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

if(BUILD_SHARED_LIBS)
    add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
else()
    set(OBJ_TARGET_NAME ${TARGET_NAME})
    add_library(${OBJ_TARGET_NAME} STATIC ${SOURCES})
endif()

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

enable_warnings_as_errors(${OBJ_TARGET_NAME})
vpux_enable_clang_format(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/level-zero/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/level-zero-ext)

if(BUILD_SHARED_LIBS)
    link_system_libraries(${OBJ_TARGET_NAME}
        PRIVATE
            IE::inference_engine
    )
else()
    link_system_libraries(${OBJ_TARGET_NAME}
        PRIVATE
            openvino::runtime
    )
endif()

link_system_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        IE::inference_engine_plugin_api
        npu_ngraph_transformations
    PUBLIC
        ze_loader
)

target_link_libraries(${OBJ_TARGET_NAME}
    PUBLIC
        npu_al
)


if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})
    if(WIN32)
        ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
        ov_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "VPU Plugin Driver Compiler Adapter")
    endif()

    add_library(${STATIC_TARGET_NAME} STATIC)
    target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
else()
    add_library(${STATIC_TARGET_NAME} ALIAS ${TARGET_NAME})
endif()

#
# targets install
#

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
