#
# Copyright 2019-2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

list(FILTER SOURCES EXCLUDE REGEX "vpux_plugin.cpp")

set(VPUX_PLUGIN_TARGET "VPUXPlugin" CACHE INTERNAL "")
set(VPUX_PLUGIN_OBJS_TARGET ${VPUX_PLUGIN_TARGET}_objs)
set(VPUX_PLUGIN_STATIC_TARGET "${VPUX_PLUGIN_TARGET}_static" CACHE INTERNAL "")
set(VPUX_PLUGIN_ENGINE_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/vpux_plugin.cpp CACHE INTERNAL "")


#
# Object library
#

add_library(${VPUX_PLUGIN_OBJS_TARGET} OBJECT ${SOURCES})

enable_warnings_as_errors(${VPUX_PLUGIN_OBJS_TARGET})
addVersionDefines(${VPUX_PLUGIN_ENGINE_SOURCE_FILE} CI_BUILD_NUMBER)
target_compile_definitions(${VPUX_PLUGIN_OBJS_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${VPUX_PLUGIN_OBJS_TARGET}
    PUBLIC
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/preproc_gapi"
)

link_system_libraries(${VPUX_PLUGIN_OBJS_TARGET}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
)

target_link_libraries(${VPUX_PLUGIN_OBJS_TARGET}
    PUBLIC
        kmb_utils
        vpux_al
    PRIVATE
        ${GAPI_LIB}
)

add_cpplint_target(${VPUX_PLUGIN_OBJS_TARGET}_cpplint FOR_TARGETS ${VPUX_PLUGIN_OBJS_TARGET})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${VPUX_PLUGIN_TARGET} FOR_TARGETS ${VPUX_PLUGIN_OBJS_TARGET} ALL)
    add_dependencies(${VPUX_PLUGIN_OBJS_TARGET} clang_format_${VPUX_PLUGIN_TARGET})
endif()

#
# Static library
#

add_library(${VPUX_PLUGIN_STATIC_TARGET} STATIC
        $<TARGET_OBJECTS:${VPUX_PLUGIN_OBJS_TARGET}>
        ${VPUX_PLUGIN_ENGINE_SOURCE_FILE})

target_compile_definitions(${VPUX_PLUGIN_STATIC_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${VPUX_DEVICE_NAME}\"")

target_include_directories(${VPUX_PLUGIN_STATIC_TARGET}
    PUBLIC
        $<TARGET_PROPERTY:${VPUX_PLUGIN_OBJS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)


link_system_libraries(${VPUX_PLUGIN_STATIC_TARGET}
    PUBLIC
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
)

target_link_libraries(${VPUX_PLUGIN_STATIC_TARGET}
    PUBLIC
        kmb_utils
    PRIVATE
        vpux_al
        ${GAPI_LIB}
)


#
# Plugin library
#

set(TARGET_NAME ${VPUX_PLUGIN_TARGET})
ie_add_plugin(NAME ${VPUX_PLUGIN_TARGET}
    DEVICE_NAME ${VPUX_DEVICE_NAME}
    SOURCES
        $<TARGET_OBJECTS:${VPUX_PLUGIN_OBJS_TARGET}>
        ${VPUX_PLUGIN_ENGINE_SOURCE_FILE}
    VERSION_DEFINES_FOR ${VPUX_PLUGIN_ENGINE_SOURCE_FILE})

target_compile_definitions(${VPUX_PLUGIN_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${VPUX_DEVICE_NAME}\"")

target_link_libraries(${VPUX_PLUGIN_TARGET}
    PRIVATE
        ${VPUX_PLUGIN_OBJS_TARGET})


install(TARGETS ${VPUX_PLUGIN_TARGET} mcmCompiler
    RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_DEVICE_NAME}
    ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_DEVICE_NAME}
    LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_DEVICE_NAME})

if(AARCH64)
    install(FILES   $<TARGET_PROPERTY:NN,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:VpualDispatcher,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:sipp_custom,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:XLink,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:RemoteFlic,IMPORTED_LOCATION_NOCONFIG>
            DESTINATION ${IE_CPACK_LIBRARY_PATH}
            COMPONENT ${VPUX_DEVICE_NAME})
endif()

set(ALL_PLUGINS)

list(APPEND ALL_PLUGINS ${VPUX_PLUGIN_TARGET})

ie_register_plugins(MAIN_TARGET ${VPUX_PLUGIN_TARGET} POSSIBLE_PLUGINS ${ALL_PLUGINS})


add_custom_target(mcm_compiler_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/config"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/mcm_config"
    COMMENT "[MCM Compiler] Copy config files"
)
add_dependencies(${VPUX_PLUGIN_TARGET} frontend_mcm mcm_compiler_copy)

if(AARCH64)
    add_custom_target(vpu_bin_copy ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpu"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/vpu.bin"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpu/"
        COMMENT "[Firmware] Copy vpu.bin"
    )
    add_custom_target(vpual_copy ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpu"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpualHostInstallPackage/lib/"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpu/"
        COMMENT "[VPUAL] Copy libraries"
    )
    add_dependencies(${VPUX_PLUGIN_TARGET} vpual_copy vpu_bin_copy)
endif()


if(ENABLE_HDDL2 AND UNIX)
        add_custom_target(pcie_drivers_copy ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/pcie_drivers"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${PCIE_DRIVERS}/"
                "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/pcie_drivers/"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config"
            COMMENT "[PCIe drivers] Copy driver packages"
        )
        add_dependencies(${VPUX_PLUGIN_TARGET} hddl2_backend pcie_drivers_copy)

    if(ENABLE_CUSTOM_HDDLUNITE)
            add_custom_target(hddlunite_components_copy ALL
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config"
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/hddlunite_artifacts/config/hddl_service.config"
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/"
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/hddlunite/hddlunite-install-script/env_host.sh"
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                COMMENT "[hddlunite components] Copy hddlunite components"
            )

            if(EXISTS "/usr/lib/libXLink.so")
                add_custom_target(hddlunite_xlink_copy ALL
                    COMMAND ${CMAKE_COMMAND} -E copy
                        "/usr/lib/libXLink.so"
                        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                    COMMENT "[hddlunite xlink] Copy hddlunite xlink"
                )
            endif()

            add_dependencies(${VPUX_PLUGIN_TARGET} hddl2_backend hddlunite_xlink_copy)
    else()
            add_custom_target(hddl_unite_copy ALL
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${HDDL_UNITE}/lib/"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${HDDL_UNITE}/thirdparty/XLink/lib/"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux_4"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${HDDL_UNITE_VPUX_4}/lib/"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux_4/"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${HDDL_UNITE_VPUX_4}/thirdparty/XLink/lib/"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux_4/"
                COMMENT "[HDDL Unite] Copy libraries"
            )

            add_dependencies(${VPUX_PLUGIN_TARGET} hddl2_backend hddl_unite_copy)
    endif()
endif()
