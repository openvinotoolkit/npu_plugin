#
# Copyright 2019-2020 Intel Corporation.
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

replace_noerror()

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

list(FILTER SOURCES EXCLUDE REGEX "vpux_plugin.cpp")

set(VPUX_PLUGIN_TARGET "openvino_intel_vpux_plugin" CACHE INTERNAL "")
set(VPUX_PLUGIN_OBJS_TARGET ${VPUX_PLUGIN_TARGET}_objs)
set(VPUX_PLUGIN_STATIC_TARGET "${VPUX_PLUGIN_TARGET}_static" CACHE INTERNAL "")
set(VPUX_PLUGIN_ENGINE_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/vpux_plugin.cpp CACHE INTERNAL "")

addVersionDefines(${VPUX_PLUGIN_ENGINE_SOURCE_FILE} CI_BUILD_NUMBER)
#
# Object library
#
if(BUILD_SHARED_LIBS)
    add_library(${VPUX_PLUGIN_OBJS_TARGET} OBJECT ${SOURCES})
endif()

#
# Plugin library
#

if(BUILD_SHARED_LIBS)
    set(VPUX_PLUGIN_TARGET_SOURCES $<TARGET_OBJECTS:${VPUX_PLUGIN_OBJS_TARGET}>)
else()
    set(VPUX_PLUGIN_TARGET_SOURCES ${SOURCES})
endif()

ie_add_plugin(NAME ${VPUX_PLUGIN_TARGET}
    DEVICE_NAME ${VPUX_DEVICE_NAME}
    SOURCES
        ${VPUX_PLUGIN_TARGET_SOURCES}
        ${VPUX_PLUGIN_ENGINE_SOURCE_FILE}
    VERSION_DEFINES_FOR ${VPUX_PLUGIN_ENGINE_SOURCE_FILE})

if(BUILD_SHARED_LIBS)
    set(TARGET_NAME ${VPUX_PLUGIN_OBJS_TARGET} CACHE INTERNAL "")
    target_link_libraries(${VPUX_PLUGIN_TARGET} PUBLIC ${VPUX_PLUGIN_OBJS_TARGET})
else()
    set(TARGET_NAME ${VPUX_PLUGIN_TARGET} CACHE INTERNAL "")
    if(ENABLE_ZEROAPI_BACKEND)
        target_link_libraries(${VPUX_PLUGIN_TARGET} INTERFACE vpux_level_zero_backend)
    endif()
    target_compile_definitions(${VPUX_PLUGIN_TARGET} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

enable_warnings_as_errors(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/preproc_gapi>
)

link_system_libraries(${TARGET_NAME}
    PUBLIC
        openvino::runtime
        openvino::runtime::dev
        vpux_utils
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        kmb_utils
        vpux_al
    PRIVATE
        vpux_utils
        preproc_gapi
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()

#
# Static library
#

if(BUILD_SHARED_LIBS)
    add_library(${VPUX_PLUGIN_STATIC_TARGET} STATIC
            $<TARGET_OBJECTS:${VPUX_PLUGIN_OBJS_TARGET}>
            ${VPUX_PLUGIN_ENGINE_SOURCE_FILE}
    )

    target_link_libraries(${VPUX_PLUGIN_STATIC_TARGET}
        PUBLIC
            ${VPUX_PLUGIN_OBJS_TARGET}
    )
else()
    add_library(${VPUX_PLUGIN_STATIC_TARGET} ALIAS ${VPUX_PLUGIN_TARGET})
endif()

if(AARCH64)
    install(FILES   $<TARGET_PROPERTY:NN,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:VpualDispatcher,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:sipp_custom,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:XLink,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:RemoteFlic,IMPORTED_LOCATION_NOCONFIG>
            DESTINATION ${IE_CPACK_LIBRARY_PATH}
            COMPONENT ${VPUX_TESTS_COMPONENT}
            EXCLUDE_FROM_ALL
    )

    # TODO: Remove duplication EISW-31024
    install(FILES   $<TARGET_PROPERTY:NN,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:VpualDispatcher,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:sipp_custom,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:XLink,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:RemoteFlic,IMPORTED_LOCATION_NOCONFIG>
            DESTINATION ${IE_CPACK_LIBRARY_PATH}
            COMPONENT tests
            EXCLUDE_FROM_ALL
    )
endif()

set(ALL_PLUGINS)

list(APPEND ALL_PLUGINS ${VPUX_PLUGIN_TARGET})

ie_register_plugins(MAIN_TARGET ${VPUX_PLUGIN_TARGET} POSSIBLE_PLUGINS ${ALL_PLUGINS})
if(WIN32)
    ie_add_api_validator_post_build_step(TARGET ${VPUX_PLUGIN_TARGET})
endif()

if(ENABLE_MCM_COMPILER)
    add_custom_target(mcm_compiler_copy ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/config"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/mcm_config"
        COMMENT "[MCM Compiler] Copy config files"
    )
    add_dependencies(${VPUX_PLUGIN_TARGET} vpux_mcm_frontend mcm_compiler_copy)
endif()

if(AARCH64)
    add_custom_target(vpu_bin_copy ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpu"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/*.bin"
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

    install(DIRECTORY   "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/"
            DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpu"
            COMPONENT    ${VPUX_TESTS_COMPONENT}
            EXCLUDE_FROM_ALL
            FILES_MATCHING
            PATTERN "*.bin")

    install(DIRECTORY   "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpualHostInstallPackage/lib/"
            DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpu"
            COMPONENT    ${VPUX_TESTS_COMPONENT}
            EXCLUDE_FROM_ALL)

    # TODO: Remove duplication EISW-31024
    install(DIRECTORY   "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/"
            DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpu"
            COMPONENT    tests
            EXCLUDE_FROM_ALL
            FILES_MATCHING
            PATTERN "*.bin")

    install(DIRECTORY   "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpualHostInstallPackage/lib/"
            DESTINATION "${IE_CPACK_RUNTIME_PATH}/vpu"
            COMPONENT    tests
            EXCLUDE_FROM_ALL)
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
    add_dependencies(${VPUX_PLUGIN_TARGET} vpux_hddl2_backend pcie_drivers_copy)

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
        add_dependencies(${VPUX_PLUGIN_TARGET} vpux_hddl2_backend hddlunite_components_copy)

        if(EXISTS "/usr/lib/libXLink.so")
            add_custom_target(hddlunite_xlink_copy ALL
                COMMAND ${CMAKE_COMMAND} -E copy
                    "/usr/lib/libXLink.so"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                COMMENT "[hddlunite xlink] Copy hddlunite xlink"
            )
            add_dependencies(${VPUX_PLUGIN_TARGET} vpux_hddl2_backend hddlunite_xlink_copy)
        endif()
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
        add_dependencies(${VPUX_PLUGIN_TARGET} vpux_hddl2_backend hddl_unite_copy)
    endif()
endif()
