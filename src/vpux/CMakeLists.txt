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

set(TARGET_NAME "vpux")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_include_directories(${TARGET_NAME}
    PUBLIC
        include)

target_link_libraries(${TARGET_NAME} PUBLIC IE::inference_engine)
