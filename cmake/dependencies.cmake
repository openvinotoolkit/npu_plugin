# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(models)
include(ExternalProject)

set_temp_directory(TEMP "${CMAKE_SOURCE_DIR}")

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=" ${MODELS_PATH})

set(DATA_PATH "${TEMP}/validation_set/src/validation_set")
debug_message(STATUS "DATA_PATH=" ${DATA_PATH})

add_models_repo(${ENABLE_MODELS} "models:inference-engine/models-ir.git")

if (ENABLE_VALIDATION_SET)
    add_lfs_repo(
        "validation_set"
        "${TEMP}/validation_set"
        "git@gitlab-icv.inn.intel.com:inference-engine/validation-set.git"
        "master"
    )
endif()

fetch_models_and_validation_set()
