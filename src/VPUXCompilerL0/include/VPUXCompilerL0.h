//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_L0_H
#define VPUX_COMPILER_L0_H

#if defined(__cplusplus)
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#if defined(__cplusplus)
#pragma once
#endif

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
#ifndef VCL_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define VCL_APICALL __cdecl
#else
#define VCL_APICALL
#endif  // defined(_WIN32)
#endif  // VCL_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef VCL_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define VCL_APIEXPORT __declspec(dllexport)
#else
#define VCL_APIEXPORT
#endif  // defined(_WIN32)
#endif  // VCL_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief Compiler handle
typedef struct __vcl_compiler_handle_t* vcl_compiler_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Executable handle
typedef struct __vcl_executable_handle_t* vcl_executable_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines version info for the VPUXCompilerL0 API
typedef struct __vcl_version_info_t {
    uint16_t major;
    uint16_t minor;

} vcl_version_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines return/error codes
typedef enum __vcl_result_t {
    VCL_RESULT_SUCCESS = 0,                             ///< [Core] success
    VCL_RESULT_ERROR_OUT_OF_MEMORY = 0x70000002,        ///< [Core] insufficient memory to satisfy call
    VCL_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,     ///< [Validation] generic error code for invalid arguments
    VCL_RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,  ///< [Validation] handle argument is not valid
    VCL_RESULT_ERROR_IO = 0x78000006,                   ///< [Core] IO error
    VCL_RESULT_ERROR_INVALID_IR = 0x78000007,           ///< [Validation] the member of modelIR is not valid
    VCL_RESULT_ERROR_UNKNOWN = 0x7ffffffe,              ///< [Core] unknown or internal error

} vcl_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines compiler properties
typedef struct __vcl_compiler_properties_t {
    const char* id;
    vcl_version_info_t version;
    uint32_t supportedOpsets;

} vcl_compiler_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines platform for compilation
typedef enum __vcl_paltform_t {
    VCL_PLATFORM_UNKNOWN = -1,

    VCL_PLATFORM_VPU3400,  ///< KMB B0 400 MHz
    VCL_PLATFORM_VPU3700,  ///< KMB B0 700 MHz
    VCL_PLATFORM_VPU3720,  ///< MTL

} vcl_paltform_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines compiler desc to be passed during creation
typedef struct __vcl_compiler_desc_t {
    vcl_paltform_t platform;
    uint32_t debug_level;

} vcl_compiler_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines log level for the compiler
typedef enum __vcl_log_level_t {
    VCL_LOG_LEVEL_NONE = 0,
    VCL_LOG_LEVEL_ERROR,
    VCL_LOG_LEVEL_WARNING,
    VCL_LOG_LEVEL_INFO,
    VCL_LOG_LEVEL_DEBUG,
    VCL_LOG_LEVEL_TRACE,

} vcl_log_level_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines compilation mode
typedef enum __vcl_compilation_mode_t {
    VCL_COMPILATION_MODE_HW = 0,
    VCL_COMPILATION_MODE_SW

} vcl_compilation_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines tensor precision
typedef enum _vcl_tensor_precision_t {
    VCL_TENSOR_PRECISION_UNKNOWN,
    VCL_TENSOR_PRECISION_FP32,
    VCL_TENSOR_PRECISION_FP16,
    VCL_TENSOR_PRECISION_UINT8 = 4,
    VCL_TENSOR_PRECISION_INT32,
    VCL_TENSOR_PRECISION_INT8 = 7,
    VCL_TENSOR_PRECISION_UINT32 = 0xD0,
} vcl_tensor_precision_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines tensor layout
typedef enum _vcl_tensor_layout_t {
    VCL_TENSOR_LAYOUT_ANY = 0x00,
    VCL_TENSOR_LAYOUT_NCHW,
    VCL_TENSOR_LAYOUT_NHWC,
    VCL_TENSOR_LAYOUT_NCDHW,
    VCL_TENSOR_LAYOUT_NDHWC,
    VCL_TENSOR_LAYOUT_C = 0x60,
    VCL_TENSOR_LAYOUT_CHW = 0x80,
    VCL_TENSOR_LAYOUT_NC = 0xC1,
    VCL_TENSOR_LAYOUT_HWC = 0xD0,
} vcl_tensor_layout_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines executable description to be passed during executable
///        creation
///
///        Format of modelIRData (defined in L0 adaptor):
///        1. API version : vcl_version_info_t
///        2. Num of data elements (now only xml + weights = 2) : uint32_t
///        3. Size of data 1 (xml) : uint64_t
///        4. Data 1 : $2 bytes
///        5. Size of data 2 (weights) : uint64_t
///        6. Data 2 : $4 bytes
typedef struct __vcl_executable_desc_t {
    const uint8_t* modelIRData;
    uint64_t modelIRSize;                    ///< Size of modelIRData
    vcl_log_level_t logLevel;                ///< LogLevel passed to compiler
    vcl_compilation_mode_t compilationMode;  ///< Compilation mode passed to compiler
    vcl_tensor_precision_t inPrc;            ///< Input data precision
    vcl_tensor_layout_t inLayout;            ///< Input data layout
    vcl_tensor_precision_t outPrc;           ///< Output data precision
    vcl_tensor_layout_t outLayout;           ///< Output data layout
    const char* options;                     ///< Compiler config options
} vcl_executable_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a compiler object and returs the compiler handle
VCL_APIEXPORT vcl_result_t VCL_APICALL vclCompilerCreate(vcl_compiler_desc_t desc, vcl_compiler_handle_t* compiler);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys the compiler
VCL_APIEXPORT vcl_result_t VCL_APICALL vclCompilerDestroy(vcl_compiler_handle_t compiler);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the compiler properties, include the version and supported_opsets
VCL_APIEXPORT vcl_result_t VCL_APICALL vclCompilerGetProperties(vcl_compiler_handle_t compiler,
                                                                vcl_compiler_properties_t* properties);

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an executable object and returns the executable handle
///  Compiles modelIRData in the executable descriptor to blob and store it in the executable.
VCL_APIEXPORT vcl_result_t VCL_APICALL vclExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                           vcl_executable_handle_t* executable);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys the executable and releases the cached blob.
VCL_APIEXPORT vcl_result_t VCL_APICALL vclExecutableDestroy(vcl_executable_handle_t executable);

///////////////////////////////////////////////////////////////////////////////
/// @brief If blobBuffer is null, the function returns the size of the blob stored in the executable.
/// Otherwise the function copies the executable cached blob to the blobBuffer provided by the caller.
VCL_APIEXPORT vcl_result_t VCL_APICALL vclExecutableGetSerializableBlob(vcl_executable_handle_t executable,
                                                                        uint8_t* blobBuffer, uint64_t* blobSize);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // VPUX_COMPILER_L0_H
