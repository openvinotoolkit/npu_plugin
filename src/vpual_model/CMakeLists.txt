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

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.h)

set(TARGET_NAME vpualModel)

add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})

add_dependencies(${TARGET_NAME} mcmCompiler)

target_include_directories(vpualModel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${MCM_SCHEMA_INCLUDE_PATH} ${FLATBUFFERRS_INCLUDE_PATH})
target_link_libraries(vpualModel PRIVATE XLink VpualDispatcher NN IE::vpu_common_lib IE::inference_engine)

set_target_properties(${TARGET_NAME} PROPERTIES
    C_VISIBILITY_PRESET       hidden
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN YES
)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
