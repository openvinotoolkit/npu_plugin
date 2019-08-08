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

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    if(NOT DEFINED $ENV{CC})
        set(COMPILER aarch64-linux-gnu-gcc)
    else()
        set(COMPILER $ENV{CC})
    endif()
    message(STATUS "COMPILER SET AS: ${COMPILER}")
    set(KEY CC=${COMPILER})
endif()

add_custom_target(build_vpusmm
        WORKING_DIRECTORY ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/vsi_cmodel/vsi_cmodel/vpusmm
        COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/lib
        COMMAND $(MAKE) ${KEY}
        COMMAND $(MAKE) DESTDIR=${CMAKE_CURRENT_SOURCE_DIR} PREFIX="" install ${KEY}
        COMMAND if [ ! -e "${CMAKE_CURRENT_SOURCE_DIR}/lib/libvpusmm.so.1" ]\; then ln -s "${CMAKE_CURRENT_SOURCE_DIR}/lib/libvpusmm.so.1.0.0" "${CMAKE_CURRENT_SOURCE_DIR}/lib/libvpusmm.so.1"\; fi )

add_library(vpusmm SHARED IMPORTED GLOBAL)

add_dependencies(vpusmm build_vpusmm)

set_target_properties(vpusmm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/vsi_cmodel/vpusmm/include/vpusmm"
        IMPORTED_LOCATION ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/vsi_cmodel/vpusmm/lib/libvpusmm.so
        IMPORTED_NO_SONAME TRUE)
