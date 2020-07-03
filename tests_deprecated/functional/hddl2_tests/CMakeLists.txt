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

set(TARGET_NAME "HDDL2FunctionalTests")

find_library(XLINK_LIBRARY
        NAMES  XLink
        HINTS "${TEMP}/hddl_unite/thirdparty/XLink/lib"
        NO_DEFAULT_PATH)

add_library(XLink SHARED IMPORTED GLOBAL)

set_target_properties(XLink PROPERTIES
        IMPORTED_LOCATION ${XLINK_LIBRARY}
        IMPORTED_NO_SONAME TRUE)

file(GLOB_RECURSE HDDL2_TESTS_SOURCES
        "*.cpp"
        )

add_executable(${TARGET_NAME} ${HDDL2_TESTS_SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
        "helpers"
        "cases"
        "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2"
        )

target_compile_options(${TARGET_NAME}
        PRIVATE
        -Wall
        -Wextra
        -Werror)

target_link_libraries(${TARGET_NAME} PRIVATE
        IE::IESharedTests
        IE::vpu_common_lib
        HddlUnite
        XLink
        kmb_utils
        )

if(ENABLE_MCM_COMPILER)
    add_dependencies(${TARGET_NAME} mcm_compiler_copy_configs)
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
endif()
