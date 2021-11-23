# Copyright (C) 2021 Intel Corporation
include_directories(${SAMPLES_JSON})
ie_add_sample(NAME VPU_Stereo_NN_IR_Generator
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS VPU_Stereo_NN_IR_Generator.h
              HEADERS json.hpp
              DEPENDENCIES IE::ie_samples_utils
              DEPENDENCIES IE::ngraph)
