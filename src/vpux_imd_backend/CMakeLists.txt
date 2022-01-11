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
    "-j8"
)

add_custom_target("${TARGET_NAME}_app_MTL" ALL
    COMMAND
        ${MAKE_EXECUTABLE} "CONFIG_FILE=.config_sim_3720xx" ${COMMON_BUILD_ARGS}
    COMMAND
        ${CMAKE_COMMAND} -E copy
            "${IMD_APP_PATH}/mvbuild/3720/InferenceManagerDemo.elf"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/IMD/3720/InferenceManagerDemo.elf"
    BYPRODUCTS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/IMD/3720/InferenceManagerDemo.elf"
    WORKING_DIRECTORY ${IMD_APP_PATH}
    COMMENT "[IMD] Build for MTL"
    VERBATIM
)
install(
    FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/IMD/3720/InferenceManagerDemo.elf"
    DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpux/IMD/3720"
    COMPONENT    ${VPUX_PLUGIN_COMPONENT}
)
add_dependencies("${TARGET_NAME}_app_MTL" "${TARGET_NAME}_app_prepare")

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

add_dependencies(${TARGET_NAME} "${TARGET_NAME}_app_MTL")

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
