#
# Copyright 2019-2020 Intel Corporation.
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

set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(CROSS_TOOLCHAIN_FLAGS_ "" CACHE STRING "" FORCE)
set(CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING "" FORCE)
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)

add_subdirectory(llvm-project/llvm EXCLUDE_FROM_ALL)

set(LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
set(LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm")

set(MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir")
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir")

set(LLVM_INCLUDE_DIRS
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
)

set(MLIR_INCLUDE_DIRS
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

#
# flatbuffers
#

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)

add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

# FIXME: this flag is forced while building in a single tree with OV
# but code cannot be compiled with the flag
if(NOT MSVC)
    target_compile_options(flatbuffers PRIVATE -Wno-error=suggest-override)
    target_compile_options(flatc PRIVATE -Wno-error=suggest-override)
endif()

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
    add_subdirectory(gapi_sipp/vpu)

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
# hddlunite
#

if(ENABLE_CUSTOM_HDDLUNITE)
    set(HDDLUNITE_BUILD_DIR "${CMAKE_BINARY_DIR}/thirdparty/hddlunite/")
    add_subdirectory(hddlunite)

    set(HDDLUNITE_CPACK_RUNTIME_PATH hddl_unite/bin)
    set(HDDLUNITE_CPACK_ARCHIVE_PATH hddl_unite/lib)
    set(HDDLUNITE_CPACK_LIBRARY_PATH hddl_unite/lib)

    if(NOT AARCH64 AND NOT WIN32)
        install(FILES
                hddlunite/common/APICommon.h
                hddlunite/common/QueryType.h
                hddlunite/api/include/ChannelUtils.h
                hddlunite/api/include/DeviceLog.h
                hddlunite/api/include/FileUtils.h
                hddlunite/api/include/HddlUnite.h
                hddlunite/api/include/HddlUniteC.h
                hddlunite/api/include/DeviceResetC.h
                hddlunite/api/include/DeviceReset.h
                hddlunite/api/include/InferBlob.h
                hddlunite/api/include/InferData.h
                hddlunite/api/include/Inference.h
                hddlunite/api/include/InferGraph.h
                hddlunite/api/include/GAPIGraph.h
                hddlunite/api/include/GAPIInvokeData.h
                hddlunite/api/include/Query.h
                hddlunite/api/include/RemoteMemory.h
                hddlunite/api/include/RemotePipeline.h
                hddlunite/api/include/WorkloadContext.h
                hddlunite/api/include/WorkloadContextC.h
                hddlunite/api/include/SetMode.h
                hddlunite/utils/Parameter.h
            DESTINATION "hddl_unite/include"
            COMPONENT hddlunite
        )

        set(HDDLUNITE_COMPONENTS
                FileSyncTool
                FileTransferTool
                GetDeviceLog
                GroupTool
                hddl_scheduler_service
                SetHDDLMode
                HddlUnite
        )
    elseif(AARCH64)
        install(FILES
                hddlunite/common/APICommon.h
                hddlunite/device_client/include/DeviceClient.h
            DESTINATION "hddl_unite/include"
            COMPONENT hddlunite
        )

        set(HDDLUNITE_COMPONENTS
                hddl_device_service
                rop_manager
        )
    endif()

    install(TARGETS ${HDDLUNITE_COMPONENTS}
            EXPORT hddluniteTargets
            RUNTIME DESTINATION ${HDDLUNITE_CPACK_RUNTIME_PATH} COMPONENT hddlunite
            ARCHIVE DESTINATION ${HDDLUNITE_CPACK_ARCHIVE_PATH} COMPONENT hddlunite
            LIBRARY DESTINATION ${HDDLUNITE_CPACK_LIBRARY_PATH} COMPONENT hddlunite
    )

    install(FILES   hddlunite_artifacts/config/hddl_service.config
            DESTINATION "hddl_unite/config"
            COMPONENT hddlunite)

    if(NOT AARCH64 AND NOT WIN32)
        set(HDDLUNITE_XLINK_LIB hddlunite/thirdparty/xlink/lib/XLink_DLL.lib)
    elseif(WIN32)
        set(HDDLUNITE_XLINK_LIB hddlunite/thirdparty/xlink/lib/XLink_DLL.dll)
    endif()
    install(FILES   ${HDDLUNITE_XLINK_LIB}
            DESTINATION "hddl_unite/thirdparty/XLink"
            COMPONENT hddlunite)

    if(NOT AARCH64 AND NOT WIN32)
        set(HDDLUNITE_ITT_LIB hddlunite/thirdparty/ittnotify/ittnotify_x86/lib/libittnotify_collector.so)
    elseif(WIN32)
        set(HDDLUNITE_ITT_LIB hddlunite/thirdparty/ittnotify/ittnotify_win/lib/ittnotify_collector.dll)
    elseif(AARCH64)
        set(HDDLUNITE_ITT_LIB hddlunite/thirdparty/ittnotify/ittnotify_arm/lib/libittnotify_collector.so)
    endif()
    install(FILES   ${HDDLUNITE_ITT_LIB}
            DESTINATION ${HDDLUNITE_CPACK_LIBRARY_PATH}
            COMPONENT hddlunite)

    if(AARCH64)
        set(HDDLUNITE_ENV hddlunite/hddlunite-install-script/env_arm.sh)
    elseif(NOT WIN32)
        set(HDDLUNITE_ENV hddlunite/hddlunite-install-script/env_host.sh)
    endif()
    install(FILES   ${HDDLUNITE_ENV}
            DESTINATION hddl_unite/
            COMPONENT hddlunite)
endif()

#
# zeroApi
#

if(ENABLE_ZEROAPI_BACKEND)
    # We have to avoid linking against loader with debug postfix due it's a part of driver
    # and the name will be the same for release and debug configurations
    set(OLD_CMAKE_DEBUG_POSTFIX CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "")
    add_subdirectory(level-zero EXCLUDE_FROM_ALL)
    set(CMAKE_DEBUG_POSTFIX OLD_CMAKE_DEBUG_POSTFIX)
endif()

#
# bitcompactor
#
set(BITCOMPACTOR_TARGET_NAME "bitcompactor")
add_library(${BITCOMPACTOR_TARGET_NAME}
    OBJECT
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/src/bitCompactor.cpp")

target_include_directories(${BITCOMPACTOR_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/include>
)
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${BITCOMPACTOR_TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()

set(HUFFMAN_CODEC_TARGET_NAME "huffman_codec")
add_library(${HUFFMAN_CODEC_TARGET_NAME}
    OBJECT
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/contrib/huffman_encoding/src/logging.cpp"
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/contrib/huffman_encoding/src/ms_vasprintf.cpp"
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/contrib/huffman_encoding/src/Huffman.cpp"
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/contrib/huffman_encoding/src/huffmanCodec.cpp")

target_include_directories(${HUFFMAN_CODEC_TARGET_NAME}
    PUBLIC
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/include/huffman_encoding"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler"
)

#
# json
#

if(ENABLE_KMB_SAMPLES)
    set(SAMPLES_JSON "${CMAKE_CURRENT_SOURCE_DIR}/json/single_include/nlohmann/" CACHE INTERNAL "")
endif()

#
# nn cost model
#
if(NOT MSVC)
    add_subdirectory(vpucostmodel)
endif()
