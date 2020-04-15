# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

configure_file(get_model_repos.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/get_model_repos.hpp @ONLY)

addIeTarget(
    NAME kmbTestData
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES ${CMAKE_CURRENT_BINARY_DIR}
    LINK_LIBRARIES IE::ieTestHelpers
    DEFINES
        INSTANTIATE_TESTS=1
        PUBLIC
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
)
