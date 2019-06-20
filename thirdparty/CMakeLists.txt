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

add_custom_target(vpusmm_target
        COMMAND make libvpusmm.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vsi_cmodel/vpusmm
        VERBATIM
        )

add_library(vpusmm SHARED IMPORTED GLOBAL)

add_dependencies(vpusmm vpusmm_target)
set_target_properties(vpusmm
        PROPERTIES
        IMPORTED_LOCATION ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/vsi_cmodel/vpusmm/libvpusmm.so
        IMPORTED_NO_SONAME 1)
