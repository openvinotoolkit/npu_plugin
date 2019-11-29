#
# Copyright 2019 Intel Corporation.
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

set(TARGET_NAME "HDDL2Plugin")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR src/hddl2_plugin.cpp
              DEVICE_NAME HDDL2)

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Werror)

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine)

if(ENABLE_KMB_CLANG_FORMAT)
    add_clang_format_target_kmb(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME})
endif()
