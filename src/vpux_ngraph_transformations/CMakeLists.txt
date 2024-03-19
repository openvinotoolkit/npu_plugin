#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "npu_ngraph_transformations")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

enable_warnings_as_errors(${TARGET_NAME})
vpux_enable_clang_format(${TARGET_NAME})

if(MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /wd4146     # unary minus operator applied to unsigned type, result still unsigned
            /wd4996     # code uses a function, class member, variable, or typedef that's marked deprecated
    )
endif()

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

link_system_libraries(${TARGET_NAME}
    PUBLIC
        openvino::reference
        openvino::runtime
        openvino::runtime::dev
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_ov_utils
        npu_llvm_utils
)

#
# targets install
#

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)

    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()
