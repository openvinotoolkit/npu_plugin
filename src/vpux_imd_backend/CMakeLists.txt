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

add_custom_target("${TARGET_NAME}_copy_app" ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/InferenceManagerDemo_vpu_2_0.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E copy
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/InferenceManagerDemo_vpu_2_7.elf"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux/InferenceManagerDemo_vpu_2_7.elf"
    COMMENT "[IMD] Copy InferenceManagerDemo application"
    VERBATIM
)
install(
    FILES
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/InferenceManagerDemo_vpu_2_0.elf"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/InferenceManagerDemo_vpu_2_7.elf"
    DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpux"
    COMPONENT    ${VPUX_PLUGIN_COMPONENT}
)

replace_noerror()

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} MODULE ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TARGET_NAME} "${TARGET_NAME}_copy_app")

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
