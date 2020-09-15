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
add_subdirectory(vpux)
add_subdirectory(vpual_backend)
if(X86_64 AND NOT WIN32)
    add_subdirectory(mcmCompiler)
endif()
add_subdirectory(frontend_mcm)
add_subdirectory(kmb_plugin)
if(ENABLE_HDDL2)
    add_subdirectory(hddl2_plugin)
endif()
add_subdirectory(vpux_plugin)
