#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

replace_noerror()
find_package(ZLIB)

set(TARGET_NAME "vpux_utils")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

ie_faster_build(${TARGET_NAME}
    PCH PUBLIC "src/precomp.hpp"
)

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
vpux_enable_clang_format(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
)

foreach(include_dir IN LISTS LLVM_INCLUDE_DIRS)
    target_include_directories(${TARGET_NAME}
        SYSTEM PUBLIC
            $<BUILD_INTERFACE:${include_dir}>
    )
endforeach()

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/generated>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# TODO It doesn't work with openvino::itt
target_compile_definitions(${TARGET_NAME} PUBLIC
        $<TARGET_PROPERTY:IE::itt,INTERFACE_COMPILE_DEFINITIONS>)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine_plugin_api
    PUBLIC
        IE::ngraph_reference
        openvino::runtime::dev
        openvino::runtime
        flatbuffers
)

add_dependencies(${TARGET_NAME}
    vpux_mlir_compiler_schema
)

add_dependencies(${TARGET_NAME}
    LLVMSupport
)

if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET_NAME} PUBLIC LLVMSupport)
else()
    bundle_static_library(${TARGET_NAME} LLVMSupport)
endif()

#
# targets install
#

# vpux_utils is a static library and is not required for shared library configuration
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()
