#
# Copyright 2020 Intel Corporation.
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
        )

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${GAPI_LIB}
            FOR_TARGETS ${GAPI_LIB}
            ALL
            )
    add_dependencies(${GAPI_LIB} clang_format_${GAPI_LIB})
endif()
