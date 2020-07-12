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

set(TARGET_NAME vpux)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

# TODO: investigate linkage error for vpux target if building shared library on Windows
if(UNIX)
    add_library(${TARGET_NAME} SHARED ${SOURCES})
else()
    add_library(${TARGET_NAME} STATIC ${SOURCES})
endif()


target_include_directories(${TARGET_NAME}
    PUBLIC
        include
    PRIVATE
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME} PUBLIC  IE::inference_engine
                                     PRIVATE IE::vpu_common_lib)

if(UNIX)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wall
            -Werror
            -Wextra)
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
