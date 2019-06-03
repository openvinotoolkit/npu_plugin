#
# Copyright 2017-2019 Intel Corporation.
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

#
# Build common part
#

add_subdirectory(graph_transformer)

add_subdirectory(
    "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius"
    "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/movidius")

#
# Build plugins
#

set(plugin_target "")

if(ENABLE_KMB)
    add_subdirectory(kmb_plugin)
endif()
