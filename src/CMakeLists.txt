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

add_subdirectory(utils)
add_subdirectory(vpux_al)
add_subdirectory(vpual_backend)
add_subdirectory(mcmCompiler)
add_subdirectory(frontend_mcm)

# TODO Should be only on AARCH64. Enabled for kmbPlugin Windows build 
#if(AARCH64)
    add_subdirectory(hddl2_plugin/preproc_gapi)
#endif()

add_subdirectory(kmb_plugin)
# TODO Rename to vpux_plugin
add_subdirectory(hddl2_plugin)
if(ENABLE_HDDL2)
    add_subdirectory(hddl2_backend)
endif()
if(ENABLE_ZEROAPI_BACKEND)
    add_subdirectory(zero_backend)
endif()
add_subdirectory(vpux_plugin)
