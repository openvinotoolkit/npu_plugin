#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

ie_add_sample(NAME VPU_Stereo_NN_IR_Generator
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS VPU_Stereo_NN_IR_Generator.h
              DEPENDENCIES IE::nlohmann_json
              DEPENDENCIES IE::ie_samples_utils
              DEPENDENCIES IE::ngraph)

