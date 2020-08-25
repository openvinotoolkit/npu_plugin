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
set(HDDL2_DEVICE_NAME HDDL2)

if(UNIX)
    set(HDDLUNITE_ARCHIVE_VERSION RELEASE_ww30)
    set(ARCH_FORMAT ".tgz")
else()
    set(HDDLUNITE_ARCHIVE_VERSION RELEASE_ww31_Windows)
    set(ARCH_FORMAT ".zip")
endif()

set_temp_directory(TEMP "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}")

include(dependency_solver)

if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
    set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
elseif(DEFINED THIRDPARTY_SERVER_PATH)
    set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
else()
    message(WARNING "Hddl unite is not found.")
endif()

if(DEFINED IE_PATH_TO_DEPS)
    reset_deps_cache(HDDL_UNITE)

    RESOLVE_DEPENDENCY(HDDL_UNITE
            ARCHIVE_LIN "hddl_unite/hddl_unite_${HDDLUNITE_ARCHIVE_VERSION}${ARCH_FORMAT}"
            ENVIRONMENT "HDDL_UNITE"
            TARGET_PATH "${TEMP}/hddl_unite")

    unset(IE_PATH_TO_DEPS)
endif()

find_library(HDDL_UNITE_LIBRARY
        NAMES  HddlUnite
        HINTS "${HDDL_UNITE}/lib"
        NO_DEFAULT_PATH)

if (WIN32)
    add_library(HddlUnite STATIC IMPORTED GLOBAL)
else()
    add_library(HddlUnite SHARED IMPORTED GLOBAL)
endif()

set_target_properties(HddlUnite PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TEMP}/hddl_unite/include"
        IMPORTED_LOCATION ${HDDL_UNITE_LIBRARY}
        IMPORTED_NO_SONAME TRUE)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR src/hddl2_plugin.cpp
              DEVICE_NAME ${HDDL2_DEVICE_NAME})

if (UNIX)
    set(EXTRA_COMPILE_OPTIONS -Wall -Wextra -Werror)
endif(UNIX)

target_compile_options(${TARGET_NAME} PRIVATE ${EXTRA_COMPILE_OPTIONS})

target_compile_definitions(${TARGET_NAME} PRIVATE -DDEVICE_NAME="${HDDL2_DEVICE_NAME}")

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

set(HDDL_UNITE_OUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${FRONTEND_VPUX_INCLUDE_DIR}"
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
        vpux)

log_rpath(HDDL_UNITE "${HDDL_UNITE_LIBRARY}")

if (ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    target_compile_options(${TARGET_NAME}_test_static PRIVATE ${EXTRA_COMPILE_OPTIONS})

    target_compile_definitions(${TARGET_NAME}_test_static PRIVATE -DDEVICE_NAME="${HDDL2_DEVICE_NAME}")

    target_include_directories(${TARGET_NAME}_test_static
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hddl_unite"
            "${IE_MAIN_SOURCE_DIR}/include"
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2"
        PRIVATE
            "${FRONTEND_VPUX_INCLUDE_DIR}")

    target_link_libraries(${TARGET_NAME}_test_static
        PUBLIC
            IE::inference_engine
            IE::vpu_common_lib
            IE::inference_engine_plugin_api
        PRIVATE
            HddlUnite
            kmb_utils
            vpux)

    if(ENABLE_CLANG_FORMAT)
        add_clang_format_target(clang_format_${TARGET_NAME}_test_static FOR_TARGETS ${TARGET_NAME}_test_static ALL)
        add_dependencies(${TARGET_NAME}_test_static clang_format_${TARGET_NAME}_test_static)
    endif()
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
