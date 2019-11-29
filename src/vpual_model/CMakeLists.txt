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
add_dependencies(${TARGET_NAME} generate-graphfile-headers)

target_include_directories(vpualModel
    SYSTEM PRIVATE
        ${VPUAL_XLINK_INCLUDES}
        ${VPUAL_NN_INCLUDES}
        ${VPUAL_DISPATCHER_INCLUDES}
        ${MCM_COMPILER_INCLUDES}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vpualModel
    PRIVATE
        IE::vpu_common_lib
        IE::inference_engine
        ${VPUAL_XLINK_LIBRARY}
        ${VPUAL_DISPATCHER_LIBRARY}
        ${VPUAL_NN_LIBRARY}
        ${MCM_COMPILER_LIBRARY})

set_target_properties(${TARGET_NAME} PROPERTIES
    C_VISIBILITY_PRESET       hidden
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN YES)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
if(ENABLE_KMB_CLANG_FORMAT)
    add_clang_format_target_kmb(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME})
endif()
