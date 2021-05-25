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


set(GAPI_LIB "preproc_gapi" CACHE INTERNAL "")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${GAPI_LIB} STATIC ${SOURCES})

enable_warnings_as_errors(${GAPI_LIB})

target_include_directories(${GAPI_LIB}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${GAPI_LIB}
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

link_system_libraries(${GAPI_LIB}
    PRIVATE
        ${VPUMGR_LIBRARY}
        IE::inference_engine
        kmb_utils
        ${GAPI_SIPP_LIBRARY}
        ${GAPI_M2I_LIBRARY}
        vpux_al
        )

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${GAPI_LIB}
            FOR_TARGETS ${GAPI_LIB}
            ALL
            )
    add_dependencies(${GAPI_LIB} clang_format_${GAPI_LIB})
endif()
