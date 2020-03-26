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

find_library(VPUSMM_LIBRARY
        NAMES vpusmm
        ONLY_CMAKE_FIND_ROOT_PATH)

if(NOT VPUSMM_LIBRARY)
    message(FATAL_ERROR "vpusmm was not found")
endif()

add_library(vpusmm SHARED IMPORTED GLOBAL)

set_target_properties(vpusmm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}"
        IMPORTED_LOCATION ${VPUSMM_LIBRARY}
        IMPORTED_NO_SONAME TRUE)
