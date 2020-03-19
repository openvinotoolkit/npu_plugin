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

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "HDDL2Plugin")

if(ENABLE_HDDL2_SIMULATOR)
set(HDDLUNITE_ARCHIVE_VERSION simulator_DEBUG-75d3-03-04-2020)
else()
set(HDDLUNITE_ARCHIVE_VERSION RELEASE_ww10)
endif()

set_temp_directory(TEMP "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}")

include(dependency_solver)

reset_deps_cache(HDDL_UNITE)

RESOLVE_DEPENDENCY(HDDL_UNITE
        ARCHIVE_LIN "hddl_unite/hddl_unite_${HDDLUNITE_ARCHIVE_VERSION}.tgz"
        ENVIRONMENT "HDDL_UNITE"
        TARGET_PATH "${TEMP}/hddl_unite")

find_library(HDDL_UNITE_LIBRARY
        NAMES  HddlUnite
        HINTS "${HDDL_UNITE}/lib"
        NO_DEFAULT_PATH)

add_library(HddlUnite SHARED IMPORTED GLOBAL)

set_target_properties(HddlUnite PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TEMP}/hddl_unite/include"
        IMPORTED_LOCATION ${HDDL_UNITE_LIBRARY}
        IMPORTED_NO_SONAME TRUE)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR src/hddl2_plugin.cpp
              DEVICE_NAME HDDL2)

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Werror)

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

set(HDDL_UNITE_OUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
file( GLOB LIB_XLINKPC_DIR ${HDDL_UNITE}/kmb_hddl/lib/libXLink* )

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${IE_FRONTEND_MCM_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hddl_unite"
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2")

target_link_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        HddlUnite
        IE::vpu_common_lib
        kmb_utils
        frontend_mcm)

log_rpath(HDDL_UNITE "${HDDL_UNITE_LIBRARY}")

if (ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    target_compile_options(${TARGET_NAME}_test_static PRIVATE ${EXTRA_COMPILE_OPTIONS})

    target_include_directories(${TARGET_NAME}_test_static
        PUBLIC
            "${IE_FRONTEND_MCM_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hddl_unite"
            "${IE_MAIN_SOURCE_DIR}/include"
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2")

    target_link_libraries(${TARGET_NAME}_test_static
        PUBLIC
            IE::inference_engine
            IE::vpu_common_lib
            IE::inference_engine_plugin_api
        PRIVATE
            HddlUnite
            kmb_utils
            frontend_mcm)

    if(ENABLE_CLANG_FORMAT)
        add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    endif()
endif()
