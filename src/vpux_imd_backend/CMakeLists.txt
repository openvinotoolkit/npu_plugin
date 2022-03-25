#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#


function(create_imd_for_platform PLATFORM IMD_CONFIG TARGET_BUILD_ARGS)
    add_custom_target("${TARGET_NAME}_app_${PLATFORM}" ALL
        COMMAND
            ${MAKE_EXECUTABLE} "CONFIG_FILE=${IMD_CONFIG}" ${COMMON_BUILD_ARGS} ${TARGET_BUILD_ARGS}
        COMMAND
            ${CMAKE_COMMAND} -E copy
            "${IMD_APP_PATH}/mvbuild/${PLATFORM}/InferenceManagerDemo.elf"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/IMD/${PLATFORM}/InferenceManagerDemo.elf"
        BYPRODUCTS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/IMD/${PLATFORM}/InferenceManagerDemo.elf"
        WORKING_DIRECTORY ${IMD_APP_PATH}
        COMMENT "[IMD] Build for MTL platform ${PLATFORM} using config ${IMD_CONFIG}"
        VERBATIM
    )
    install(
        FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/IMD/${PLATFORM}/InferenceManagerDemo.elf"
        DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpux/IMD/${PLATFORM}"
        COMPONENT    ${VPUX_PLUGIN_COMPONENT}
    )
    add_dependencies("${TARGET_NAME}_app_${PLATFORM}" "${TARGET_NAME}_app_prepare")
endfunction(create_imd_for_platform)

set(TARGET_NAME "vpux_imd_backend")

#
# InferenceManagerDemo application
#

find_path (VPU_FIRMWARE_SOURCES_PATH "mdkApplication.mk")
if(NOT VPU_FIRMWARE_SOURCES_PATH)
    message(FATAL_ERROR "VPU_FIRMWARE_SOURCES_PATH is not defined")
endif()

if(NOT DEFINED ENV{MV_TOOLS_DIR} OR NOT EXISTS $ENV{MV_TOOLS_DIR})
    message(FATAL_ERROR "MV_TOOLS_DIR env var is not defined")
endif()

find_program (MAKE_EXECUTABLE "make")
if(NOT MAKE_EXECUTABLE)
    message(FATAL_ERROR "Can't find GNU make program")
endif()

set(IMD_APP_PATH "${VPU_FIRMWARE_SOURCES_PATH}/application/demo/InferenceManagerDemo")

add_custom_target("${TARGET_NAME}_app_prepare"
    COMMAND ${MAKE_EXECUTABLE} "prepare-kconfig"
    COMMAND ${MAKE_EXECUTABLE} "getTools"
    WORKING_DIRECTORY ${VPU_FIRMWARE_SOURCES_PATH}
    COMMENT "[IMD] Prepare build system"
    VERBATIM
)

set(COMMON_BUILD_ARGS
    "CONFIG_NN_LOG_VERBOSITY_LRT_WARN=y"
    "CONFIG_NN_LOG_VERBOSITY_LRT_INFO=n"
    "CONFIG_NN_LOG_VERBOSITY_LNN_WARN=y"
    "CONFIG_NN_LOG_VERBOSITY_LNN_INFO=n"
    "CONFIG_NN_LOG_VERBOSITY_SNN_WARN=y"
    "CONFIG_NN_LOG_VERBOSITY_SNN_INFO=n"
    "CONFIG_PROFILING_MASK=0b00000000"
    # E#-35030 - race condition of the threads that are performing the build
    "-j1"
)

create_imd_for_platform("3720" ".config_sim_3720xx" "")
create_imd_for_platform("ma2490" ".config" "SOC_REVISION=B0")

#
# Backend target
#

replace_noerror()

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} MODULE ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TARGET_NAME} "${TARGET_NAME}_app_3720" "${TARGET_NAME}_app_ma2490")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)

vpux_enable_clang_format(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_plugin_api
        LLVMSupport
)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        vpux_al
        vpux_utils
)

install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH}
    COMPONENT ${VPUX_PLUGIN_COMPONENT}
)
