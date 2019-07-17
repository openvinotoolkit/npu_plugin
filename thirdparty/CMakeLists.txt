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

add_subdirectory(vsi_cmodel/vpusmm)

if (ENABLE_VPUAL)
    if (NOT vpualHost_FOUND)
        if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/movidius/vpualHost")
            message(FATAL_ERROR "vpualHost is not found. Please use: git submodule update --init --recursive to get it")
        endif()

        # including vpualHost/host directory directly not to add vpualHost's examples to the build
        add_subdirectory(movidius/vpualHost/host)

        # TODO: remove some day when all warnings are fixed
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set_target_properties(XLink VpualDispatcher sipp_custom RemoteFlic NN PROPERTIES COMPILE_OPTIONS "-Wno-error")
        endif()
    endif()

    # copy memory manager library to target directory when vpualHost is enabled
    get_target_property(VPUSMM_LIB_PATH vpusmm IMPORTED_LOCATION)
    set(VPUSMM_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")
    add_custom_target(vpusmm_copy_library ALL
        COMMAND "${CMAKE_COMMAND}" -E copy "${VPUSMM_LIB_PATH}" "${VPUSMM_OUT_DIR}"
        COMMENT "[VPUSMM] Copy ${VPUSMM_LIB_PATH} to build results")

    if (ENABLE_TESTS)
        add_subdirectory(tests/integration)
    endif()
endif()

