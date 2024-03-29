#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME "npu_imd_backend")
find_package(MLIR REQUIRED CONFIG)

add_custom_target("${TARGET_NAME}_copy_app" ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simulator/InferenceManagerDemo_vpu_2_0.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/simulator/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/fpga/InferenceManagerDemo_vpu_2_0.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/fpga/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/silicon/InferenceManagerDemo_vpu_2_0.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/silicon/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simulator/InferenceManagerDemo_vpu_2_7.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/simulator/InferenceManagerDemo_vpu_2_7.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/fpga/InferenceManagerDemo_vpu_2_7.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/fpga/InferenceManagerDemo_vpu_2_7.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/silicon/InferenceManagerDemo_vpu_2_7.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/silicon/InferenceManagerDemo_vpu_2_7.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simics/InferenceManagerDemo_vpu_4.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/simics/InferenceManagerDemo_vpu_4.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simics/InferenceManagerDemo_vpu_5.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/simics/InferenceManagerDemo_vpu_5.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/fpga/InferenceManagerDemo_vpu_4.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/fpga/InferenceManagerDemo_vpu_4.elf"
    COMMENT "[IMD] Copy InferenceManagerDemo application"
    VERBATIM
)
install(
    FILES
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simulator/InferenceManagerDemo_vpu_2_0.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/fpga/InferenceManagerDemo_vpu_2_0.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/silicon/InferenceManagerDemo_vpu_2_0.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simulator/InferenceManagerDemo_vpu_2_7.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/fpga/InferenceManagerDemo_vpu_2_7.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/silicon/InferenceManagerDemo_vpu_2_7.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/simics/InferenceManagerDemo_vpu_4.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/fpga/InferenceManagerDemo_vpu_4.elf"
    DESTINATION "${OV_CPACK_RUNTIMEDIR}/vpux"
    COMPONENT    ${VPUX_PLUGIN_COMPONENT}
)


if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} MODULE ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TARGET_NAME} "${TARGET_NAME}_copy_app")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

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
        openvino::runtime::dev
        openvino::runtime
        LLVMSupport
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        npu_al
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
)

install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR}
    COMPONENT ${VPUX_PLUGIN_COMPONENT}
)
