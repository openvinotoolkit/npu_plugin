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

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "kmb_utils")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
endif()

add_library(${TARGET_NAME} STATIC ${SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE ${VPUSMM_LIBRARY} PUBLIC IE::inference_engine_lp_transformations)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
    SYSTEM PRIVATE
        ${MCM_COMPILER_INCLUDES})

# TODO: choose better MAIN_TARGET?
ie_register_plugins(
    MAIN_TARGET
        kmb_utils
    POSSIBLE_PLUGINS
        vpuPlugin kmbPlugin HDDL2Plugin)
