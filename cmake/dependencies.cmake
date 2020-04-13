# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(models)
include(ExternalProject)

set_temp_directory(TEMP "${CMAKE_SOURCE_DIR}")
debug_message(STATUS "MODELS_PATH=" ${MODELS_PATH})

if (ENABLE_MODELS)
    set(MODELS_PATH "${TEMP}/models/src")
    add_models_repo(${ENABLE_MODELS} "models:inference-engine/models-ir.git")
endif()

if (ENABLE_VALIDATION_SET)
    set(DATA_PATH "${TEMP}/validation_set/src/validation_set")
    add_lfs_repo(
            "validation_set"
            "${TEMP}/validation_set"
            "git@gitlab-icv.inn.intel.com:inference-engine/validation-set.git"
            "${MODELS_BRANCH}")
endif()

fetch_models_and_validation_set()
