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

set(IE_MAIN_KMB_PLUGIN_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(unit/kmb_plugin)

if(ENABLE_HDDL2 AND ENABLE_HDDL2_TESTS)
    add_subdirectory(unit/hddl2_tests)
endif()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if (ENABLE_BEH_TESTS)
    add_subdirectory(behavior)
endif()

if (ARM OR AARCH64)
    add_subdirectory(gapi_preproc)
endif()
