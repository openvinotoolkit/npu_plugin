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

set(TARGET_NAME "zero_compiler_adapter")
set(OBJ_TARGET_NAME "zero_compiler_adapter_obj")
set(STATIC_TARGET_NAME "zero_compiler_adapter_static")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})

ie_faster_build(${OBJ_TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

if(AARCH64)
    set(ARCH_SUBFOLDER aarch64)
else()
    set(ARCH_SUBFOLDER intel64)
endif()

# Copy VPUXCompilerL0 lib from artifacts
#  add_custom_target(VPUXCompilerL0_lib_copy ALL
#      COMMAND ${CMAKE_COMMAND} -E copy
#          "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/VPUXCompilerL0/${ARCH_SUBFOLDER}/*.dll"
#          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/"
#      COMMENT "[ZeroCompilerAdapter] Copy VPUXCompilerL0 library."
#  )
#  add_dependencies(${OBJ_TARGET_NAME} VPUXCompilerL0_lib_copy)

# if(AARCH64)
# FIXME Finding library for ARM require different approach. Use direct link for now
    # set(VPUX_COMPILER_L0_PATH "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/VPUXCompilerL0/${ARCH_SUBFOLDER}/libVPUXCompilerL0.so")
    # set(VPUX_COMPILER_L0_PATH "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/VPUXCompilerL0/${ARCH_SUBFOLDER}/VPUXCompilerL0.dll")
# else()
# find_library(VPUX_COMPILER_L0_LIBRARY
#     NAMES VPUXCompilerL0
#     HINTS "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/VPUXCompilerL0/${ARCH_SUBFOLDER}/"
#     REQUIRED
#     NO_DEFAULT_PATH)
    
#     log_rpath(VPUX_COMPILER_L0 "${VPUX_COMPILER_L0_LIBRARY}")

#     add_library(VPUXCompilerL0 SHARED IMPORTED GLOBAL)
#     add_dependencies(${OBJ_TARGET_NAME} VPUXCompilerL0)
#     set_target_properties(VPUXCompilerL0 PROPERTIES
#         IMPORTED_LOCATION ${VPUX_COMPILER_L0_LIBRARY}
#     )
# endif()

enable_warnings_as_errors(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/VPUXCompilerL0
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/level-zero/include)

link_system_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
        ze_loader)

# if(AARCH64)
target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        # ${VPUX_COMPILER_L0_PATH}
        vpux_al
        kmb_utils)
# else()
# target_link_libraries(${OBJ_TARGET_NAME}
#     PRIVATE
#         VPUXCompilerL0
#         vpux_al
#         kmb_utils)
# endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${OBJ_TARGET_NAME} FOR_TARGETS ${OBJ_TARGET_NAME} ALL)
    add_dependencies(${OBJ_TARGET_NAME} clang_format_${OBJ_TARGET_NAME})
endif()

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
