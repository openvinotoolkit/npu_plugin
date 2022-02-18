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

# if(DEFINED MV_TOOLS_DIR)
#     message("MV_TOOLS_DIR = ${MV_TOOLS_DIR}")
# else()
#     message(WARNING "MV_TOOLS_DIR is not defined. The vpux-jit will not be built.")
#     return()
# endif()

# if(NOT DEFINED MV_TOOLS_VERSION)
#     set(MV_TOOLS_VERSION "21.09.5-internal")
# endif()
# message("MV_TOOLS_VERSION = ${MV_TOOLS_VERSION}")

# set(MV_TOOLS_PATH "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}")

if(NOT ENABLE_EMULATOR)
    message(WARNING "Emulator not enabled. VPUX-JIT will not be built")
endif()

set(TARGET_NAME "vpux-jit")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_executable(${TARGET_NAME} ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tools")

#FIXME: we will currently link with emulator, and some extra hack-includes to be able to use the movisim binding
#utilities. TODO [JIRA_TASK] to move the movisim bindings into core kmb-plugin utilities
set(EMU_ROOT_DIR ${PROJECT_SOURCE_DIR}/src/vpux_emulator)
target_include_directories(
    ${TARGET_NAME}
    PUBLIC
    ${EMU_ROOT_DIR}/include/emu
    ${EMU_ROOT_DIR}/src
    ${MV_TOOLS_PATH}/common/moviSim/includes
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        vpux_elf
        vpux_utils
        vpux_mlir_compiler_static
        vpux_translate_utils_static
        vpux_emulator
)

add_custom_target(vpux_jit_copy ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux_jit"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_LIST_DIR}/VPUX_JIT_FW.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux_jit"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_LIST_DIR}/VPUX_JIT_FW_noPrints.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux_jit"
    COMMENT "[VPUX-JIT] Copy vpux-jit VPU firmware"
)

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
vpux_enable_clang_format(${TARGET_NAME})
