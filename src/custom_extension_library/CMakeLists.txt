# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME "custom_extension_library")
set(SRC sample_ext.cpp add_with_offset_op.cpp)

add_library(${TARGET_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)

link_system_libraries(${TARGET_NAME} PUBLIC IE::ngraph IE::inference_engine)
