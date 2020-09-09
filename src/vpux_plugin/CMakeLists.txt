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

set(VPUX_PLUGIN_TARGET "VPUXPlugin" CACHE INTERNAL "")

if(ENABLE_HDDL2)
    set(VPUX_PLUGIN_OBJS_TARGET ${HDDL2_PLUGIN_OBJS_TARGET})
    set(VPUX_PLUGIN_ENGINE_SOURCE_FILE ${HDDL2_PLUGIN_ENGINE_SOURCE_FILE})
else()
    set(VPUX_PLUGIN_OBJS_TARGET ${KMB_PLUGIN_OBJS_TARGET})
    set(VPUX_PLUGIN_ENGINE_SOURCE_FILE ${KMB_PLUGIN_ENGINE_SOURCE_FILE})
endif()

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

install(TARGETS ${VPUX_PLUGIN_TARGET}
    RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_DEVICE_NAME}
    ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_DEVICE_NAME}
    LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_DEVICE_NAME})

if(X86_64 AND NOT WIN32)
    install(FILES   $<TARGET_PROPERTY:cm,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:metamodel,IMPORTED_LOCATION_NOCONFIG>
                    $<TARGET_PROPERTY:model,IMPORTED_LOCATION_NOCONFIG>
            DESTINATION ${IE_CPACK_LIBRARY_PATH}
            COMPONENT ${VPUX_DEVICE_NAME})
endif()

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
list(APPEND ALL_PLUGINS ${KMB_PLUGIN_TARGET})
if(ENABLE_HDDL2)
    list(APPEND ALL_PLUGINS ${HDDL2_PLUGIN_TARGET})
endif()
list(APPEND ALL_PLUGINS ${VPUX_PLUGIN_TARGET})

ie_register_plugins(MAIN_TARGET ${VPUX_PLUGIN_TARGET} POSSIBLE_PLUGINS ${ALL_PLUGINS})
