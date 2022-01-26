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

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "preproc_gapi")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_warnings_as_errors(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        ${VPUMGR_LIBRARY}
        openvino::runtime
        IE::inference_engine_plugin_api
        kmb_utils
        ${GAPI_SIPP_LIBRARY}
        ${GAPI_M2I_LIBRARY}
        vpux_al
)

if(BUILD_SHARED_LIBS)
    link_system_libraries(${TARGET_NAME}
        PRIVATE
            IE::inference_engine
    )
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME}
        FOR_TARGETS ${TARGET_NAME}
        ALL
    )
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()

#
# targets install
#

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()
