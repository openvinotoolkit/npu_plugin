#
# Copyright 2019-2020 Intel Corporation.
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

#
# Replace compiler flags
#

foreach(flag IN ITEMS "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
    string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()

set(BUILD_SHARED_LIBS OFF)

#
# LLVM/MLIR
#

if(ENABLE_EXPERIMENTAL_MLIR)
    set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_BACKTRACES ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_CRASH_OVERRIDES ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
    set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
    set(CROSS_TOOLCHAIN_FLAGS_ "" CACHE STRING "" FORCE)
    set(CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING "" FORCE)
    add_subdirectory(llvm-project/llvm EXCLUDE_FROM_ALL)
endif()

#
# flatbuffers
#

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

vpux_add_native_tool(flatc "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers"
    CMAKE_ARGS
        "FLATBUFFERS_BUILD_TESTS:BOOL=OFF"
        "FLATBUFFERS_INSTALL:BOOL=OFF"
        "FLATBUFFERS_BUILD_FLATC:BOOL=ON"
)

#
# vpual
#

if(AARCH64)
    find_package(vpualHost REQUIRED CONFIG NO_DEFAULT_PATH
        PATHS "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpualHostInstallPackage/share/vpualHost"
        CMAKE_FIND_ROOT_PATH_BOTH)

    set_target_properties(NN PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(VpualDispatcher PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(sipp_custom PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(XLink PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(RemoteFlic PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(vpumgr PROPERTIES IMPORTED_GLOBAL TRUE)

    add_subdirectory(gapi_sipp/sipp)
    add_subdirectory(gapi_sipp/m2i)

    if(ENABLE_TESTS)
        add_subdirectory(tests/integration)
    endif()

    set(VPUAL_XLINK_LIBRARY XLink CACHE INTERNAL "")
    set(VPUAL_XLINK_INCLUDES $<TARGET_PROPERTY:XLink,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUAL_DISPATCHER_LIBRARY VpualDispatcher CACHE INTERNAL "")
    set(VPUAL_DISPATCHER_INCLUDES $<TARGET_PROPERTY:VpualDispatcher,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUAL_NN_LIBRARY NN CACHE INTERNAL "")
    set(VPUAL_NN_INCLUDES $<TARGET_PROPERTY:NN,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUMGR_LIBRARY vpumgr CACHE INTERNAL "")
    set(VPUMGR_INCLUDES $<TARGET_PROPERTY:vpumgr,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(GAPI_SIPP_LIBRARY gapi_sipp CACHE INTERNAL "")
    set(GAPI_SIPP_INCLUDES $<TARGET_PROPERTY:gapi_sipp,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(GAPI_M2I_LIBRARY gapi_m2i CACHE INTERNAL "")
    set(GAPI_M2I_INCLUDES $<TARGET_PROPERTY:gapi_m2i,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")
else()
    set(VPUAL_XLINK_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_XLINK_INCLUDES "" CACHE INTERNAL "")

    set(VPUAL_DISPATCHER_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_DISPATCHER_INCLUDES "" CACHE INTERNAL "")

    set(VPUAL_NN_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_NN_INCLUDES "" CACHE INTERNAL "")

    set(GAPI_SIPP_LIBRARY "" CACHE INTERNAL "")
    set(GAPI_SIPP_INCLUDES "" CACHE INTERNAL "")

    set(GAPI_M2I_LIBRARY "" CACHE INTERNAL "")
    set(GAPI_M2I_INCLUDES "" CACHE INTERNAL "")
endif()

#
# zeroApi
#

if(ENABLE_ZEROAPI_BACKEND)
    add_library(zeroApi STATIC IMPORTED GLOBAL)
    set_target_properties(zeroApi PROPERTIES IMPORTED_LOCATION ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/zePackage/lib/ze_loader_vpu64.lib)
    set_target_properties(zeroApi PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/zePackage/inc)
endif()

