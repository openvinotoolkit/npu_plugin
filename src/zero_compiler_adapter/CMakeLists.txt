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

# Copy umd_converter lib from artifacts
 add_custom_target(umd_converter_lib_copy ALL
     COMMAND ${CMAKE_COMMAND} -E copy
         "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/umd_converter/${ARCH_SUBFOLDER}/*.so"
         "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/"
     COMMENT "[ZeroCompilerAdapter] Copy umb converter library."
 )
 add_dependencies(${OBJ_TARGET_NAME} umd_converter_lib_copy)

if(AARCH64)
# FIXME Finding library for ARM require different approach. Use direct link for now
    set(UMD_CONVERTER_PATH "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/umd_converter/${ARCH_SUBFOLDER}/libumd_converter.so")
else()
find_library(UMD_CONVERTER_LIBRARY
    NAMES umd_converter
    HINTS "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/umd_converter/${ARCH_SUBFOLDER}/"
    REQUIRED
    NO_DEFAULT_PATH)
    
    log_rpath(UMD_CONVERTER "${UMD_CONVERTER_LIBRARY}")

    add_library(umd_converter SHARED IMPORTED GLOBAL)
    add_dependencies(${OBJ_TARGET_NAME} umd_converter)
    set_target_properties(umd_converter PROPERTIES
        IMPORTED_LOCATION ${UMD_CONVERTER_LIBRARY}
    )
endif()

enable_warnings_as_errors(${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/umd_converter)

link_system_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib)

if(AARCH64)
target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        ${UMD_CONVERTER_PATH}
        vpux_al
        kmb_utils)
else()
target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        umd_converter
        vpux_al
        kmb_utils)
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${OBJ_TARGET_NAME} FOR_TARGETS ${OBJ_TARGET_NAME} ALL)
    add_dependencies(${OBJ_TARGET_NAME} clang_format_${OBJ_TARGET_NAME})
endif()

add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})

add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
