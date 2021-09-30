# Copyright (C) 2021 Intel Corporation

ie_add_sample(NAME ngraph_stereo
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES IE::ie_samples_utils
              DEPENDENCIES IE::ngraph)
