#
# Copyright 2020-2021 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(STATIC_TARGET_NAME "vpux_ngraph_transformations")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${STATIC_TARGET_NAME} STATIC ${SOURCES})

ie_faster_build(${STATIC_TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

enable_warnings_as_errors(${STATIC_TARGET_NAME})

if(MSVC)
    target_compile_options(${STATIC_TARGET_NAME}
        PRIVATE
            /wd4146     # unary minus operator applied to unsigned type, result still unsigned
            /wd4996     # code uses a function, class member, variable, or typedef that's marked deprecated
    )
endif()

target_include_directories(${STATIC_TARGET_NAME}
    PUBLIC
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

link_system_libraries(${STATIC_TARGET_NAME}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_legacy
        IE::inference_engine_transformations
        IE::ngraph_reference
)
target_link_libraries(${STATIC_TARGET_NAME}
    PUBLIC
        kmb_utils
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${STATIC_TARGET_NAME}
        FOR_TARGETS ${STATIC_TARGET_NAME}
        EXCLUDE_PATTERNS ".*precomp.hpp"
        ALL
    )
    add_dependencies(${STATIC_TARGET_NAME} clang_format_${STATIC_TARGET_NAME})
endif()
