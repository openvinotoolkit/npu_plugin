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

set(TARGET_NAME "HDDL2Plugin")

# downloading HDDL_UNITE archive
set(ARCHIVE_VERSION ww52)
set_temp_directory(TEMP "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}")

include(dependency_solver)

reset_deps_cache(HDDL_UNITE_DIR HDDL_UNITE_LIBRARY)

RESOLVE_DEPENDENCY(HDDL_UNITE_DIR
        ARCHIVE_LIN "hddl_unite/hddl_unite_${ARCHIVE_VERSION}.tgz"
        ENVIRONMENT "HDDL_UNITE_DIR"
        TARGET_PATH "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/temp/hddl_unite")

find_library(HDDL_UNITE_LIBRARY
        NAMES  HddlUnite
        HINTS "${HDDL_UNITE_DIR}/kmb_hddl/lib")

add_library(HddlUnite SHARED IMPORTED GLOBAL)

set_target_properties(HddlUnite PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/temp/hddl_unite/kmb_hddl/include"
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
file( GLOB LIB_XLINKPC_DIR ${HDDL_UNITE_DIR}/kmb_hddl/lib/libXLinkPC* )

add_custom_target(hddl_unite_copy_library
        COMMAND "${CMAKE_COMMAND}" -E copy ${LIB_XLINKPC_DIR} ${HDDL_UNITE_OUT_DIR}
        COMMENT "[HDDL_UNITE] Copy hddl unite library to build results")

add_dependencies(${TARGET_NAME} hddl_unite_copy_library)

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        HddlUnite)


add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

target_compile_options(${TARGET_NAME}_test_static PRIVATE ${EXTRA_COMPILE_OPTIONS})

target_include_directories(${TARGET_NAME}_test_static
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME}_test_static
    PUBLIC
        IE::inference_engine
        IE::inference_engine_plugin_api
    PRIVATE
        HddlUnite)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
endif()
