#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

cmake_policy(SET CMP0054 NEW)

include(ExternalProject)

if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if(NOT BUILD_SHARED_LIBS)
    set(TEMP "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/temp")
else()
    set_temp_directory(TEMP "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}")
endif()

# FIXME: Create empty file to avoid errors on CI
file(TOUCH "${CMAKE_BINARY_DIR}/ld_library_rpath_64.txt")

#
# Models and Images for tests
#

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=${MODELS_PATH}")

set(DATA_PATH "${TEMP}/validation_set/src/validation_set")
debug_message(STATUS "DATA_PATH=${DATA_PATH}")

if (MODELS_REPO)
    debug_message(STATUS "MODELS_REPO=${MODELS_REPO}")
    add_models_repo(${ENABLE_MODELS} ${MODELS_REPO})
endif()

# TODO move it out into submodules
if (ENABLE_VALIDATION_SET)
    add_lfs_repo(
        "validation_set"
        "${TEMP}/validation_set"
        "git@gitlab-icv.inn.intel.com:inference-engine/validation-set.git"
        "master"
    )
endif()

fetch_models_and_validation_set()

if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ")
endif()
