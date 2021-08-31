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

#ifndef GRAPH_COMPILER_API_H
#define GRAPH_COMPILER_API_H

#include <stdint.h>

// clang-format off
#if defined(__cplusplus)
#pragma once
#endif

#if defined(__cplusplus)
extern "C" {
#endif

    ///////////////////////////////////////////////////////////////////////////////
#ifndef GC_APICALL
#if defined(_WIN32)
    /// @brief Calling convention for all API functions
#define GC_APICALL __cdecl
#else
#define GC_APICALL
#endif // defined(_WIN32)
#endif // GC_APICALL

#ifdef GC_DLL_EXPORTS

    ///////////////////////////////////////////////////////////////////////////////
#ifndef GC_DLLEXPORT
#if defined(_WIN32)
    /// @brief Microsoft-specific dllexport storage-class attribute
#define GC_DLLEXPORT  __declspec(dllexport)
#endif // defined(_WIN32)
#endif // GC_DLLEXPORT

    ///////////////////////////////////////////////////////////////////////////////
#ifndef GC_DLLEXPORT
#if __GNUC__ >= 4
    /// @brief GCC-specific dllexport storage-class attribute
#define GC_DLLEXPORT  __attribute__ ((visibility ("default")))
#endif // __GNUC__ >= 4
#endif // GC_DLLEXPORT

#else // GC_DLL_EXPORTS
#define GC_DLLEXPORT
#endif // GC_DLL_EXPORTS


    typedef struct _gc_compiler_handle_t *gc_compiler_handle_t;
    typedef struct _gc_executable_handle_t *gc_executable_handle_t;
    typedef uint64_t gc_va;

    typedef enum _gc_result_t
    {
        GC_RESULT_SUCCESS = 0,                             ///< [Core] success
        GC_RESULT_ERROR_OUT_OF_MEMORY = 0x70000002,        ///< [Core] insufficient memory to satisfy call
        GC_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,     ///< [Validation] generic error code for invalid arguments
        GC_RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,  ///< [Validation] handle argument is not valid
        GC_RESULT_ERROR_UNKNOWN = 0x7ffffffe,              ///< [Core] unknown or internal error
    } gc_result_t;


    typedef struct _gc_version_info_t
    {
        struct
        {
            uint16_t major;
            uint16_t minor;
        };
    } gc_version_info_t;

    typedef enum _gc_executable_input_type_t
    {
        GC_EXECUTABLE_INPUT_TYPE_NATIVE = 0x1,
        GC_EXECUTABLE_INPUT_TYPE_NGRAPH_LITE  = 0x2,
        } gc_executable_input_type_t;

    typedef enum _gc_executable_opset_type_t
    {
        GC_EXECUTABLE_OPSET_TYPE_OV6 = 0x1,
        GC_EXECUTABLE_OPSET_TYPE_OV7 = 0x2,
        } gc_executable_opset_type_t;

    typedef struct _gc_compiler_properties_t
    {
        gc_version_info_t compiler_version;
        gc_executable_input_type_t supported_formats;
        gc_executable_opset_type_t supported_opsets;
    } gc_compiler_properties_t;

    typedef enum _product_family_t
    {
        GC_PRODUCT_FAMILY_UNKNOWN      = 0,
        GC_PRODUCT_FAMILY_MYRIADX,
        GC_PRODUCT_FAMILY_KEEMBAY,
        GC_PRODUCT_FAMILY_METEORLAKE,
        GC_PRODUCT_FAMILY_LUNARLAKE,
        } product_family_t;

    typedef enum _revision_id_t
    {
        GC_REVISION_ID_A0 = 0,
        GC_REVISION_ID_A1,
        GC_REVISION_ID_A3,
        GC_REVISION_ID_B,
        GC_REVISION_ID_C,
        GC_REVISION_ID_D,
        GC_REVISION_ID_K,
        } revision_id_t;

    typedef struct _gc_compiler_desc_t
    {
        product_family_t family;
        revision_id_t revision_id;
        uint32_t debug_level;
    } gc_compiler_desc_t;

    typedef enum _gc_buffer_type_t
    {
        GC_BUFFER_TYPE_UNKNOWN = 0,
        GC_BUFFER_TYPE_SCRATCH,
        GC_BUFFER_TYPE_TENSOR,
        GC_BUFFER_TYPE_PROFILING,
        GC_BUFFER_TYPE_WEIGHTS,
        GC_BUFFER_TYPE_KERNELS,
        GC_BUFFER_TYPE_SCHEDULE,
        } gc_buffer_type_t;

    typedef enum _gc_tensor_type_t
    {
        GC_BUFFER_TYPE_INPUT,
        GC_BUFFER_TYPE_OUTPUT,
        } gc_tensor_type_t;

#ifndef GC_MAX_TENSOR_NAME
    /// @brief Maximum session name string size
    #define GC_MAX_TENSOR_NAME  256
#endif // GC_MAX_TENSOR_NAME

typedef enum _gc_tensor_dimension_t
{
    GC_TENSOR_DIMENSION_N,
    GC_TENSOR_DIMENSION_C,
    GC_TENSOR_DIMENSION_H,
    GC_TENSOR_DIMENSION_W,
    GC_TENSOR_DIMENSION_D,
    GC_TENSOR_DIMENSION_MAX,
    } gc_tensor_dimension_t;

typedef enum _gc_tensor_precision_t
{
    GC_TENSOR_PRECISION_UNKNOWN,
    GC_TENSOR_PRECISION_FP32,
    GC_TENSOR_PRECISION_FP16,
    GC_TENSOR_PRECISION_UINT16,
    GC_TENSOR_PRECISION_UINT8,
    GC_TENSOR_PRECISION_INT32,
    GC_TENSOR_PRECISION_INT16,
    GC_TENSOR_PRECISION_INT8,
    GC_TENSOR_PRECISION_INT4,
    GC_TENSOR_PRECISION_INT2,
    GC_TENSOR_PRECISION_BIN,
    } gc_tensor_precision_t;

typedef enum _gc_tensor_layout_t
{
    GC_TENSOR_LAYOUT_ANY        = 0x00,
    GC_TENSOR_LAYOUT_NCHW,
    GC_TENSOR_LAYOUT_NHWC,
    GC_TENSOR_LAYOUT_NCDHW,
    GC_TENSOR_LAYOUT_NDHWC,
    GC_TENSOR_LAYOUT_OIHW       = 0x40,
    GC_TENSOR_LAYOUT_C          = 0x60,
    GC_TENSOR_LAYOUT_CHW        = 0x80,
    GC_TENSOR_LAYOUT_HW         = 0xC0,
    GC_TENSOR_LAYOUT_NC,
    GC_TENSOR_LAYOUT_CN,
    GC_TENSOR_LAYOUT_BLOCKED    = 0xC8
} gc_tensor_layout_t;

typedef struct _gc_tensor_attributes_t
{
    char name[GC_MAX_TENSOR_NAME];
    gc_tensor_type_t type;
    uint32_t dims[GC_TENSOR_DIMENSION_MAX];
    uint32_t strides[GC_TENSOR_DIMENSION_MAX + 1];
    gc_tensor_precision_t precision;
    gc_tensor_layout_t layout;
} gc_tensor_attributes_t;

typedef struct _gc_buffer_properties_t
{
    gc_buffer_type_t type;
    uint32_t size;                             // Buffer size in bytes
    gc_tensor_attributes_t *tensor_attributes; // NULL for linear buffers
} gc_buffer_properties_t;

typedef struct _gc_executable_desc_t
{
    const char *options;   // Includes profiling and other options
    const uint8_t *buffer; // Will this be the serialized or deserialized form? Having serialized will have a performance and memory impact
    uint32_t buffer_size;
    gc_executable_input_type_t type;
} gc_executable_desc_t;

typedef struct _gc_initialize_schedule_desc_t
{
    gc_va kernel;
    gc_va schedule;
    gc_va scratch;
    gc_va weights;
    uint32_t kernel_size;
    uint8_t *kernel_data;
    uint32_t schedule_size;
    uint8_t *schedule_data;
    uint32_t weights_size;
    uint8_t *weights_data;
} gc_initialize_schedule_desc_t;

GC_DLLEXPORT gc_result_t GC_APICALL
CompilerCreate(gc_compiler_desc_t desc, gc_compiler_handle_t *compiler);

GC_DLLEXPORT gc_result_t GC_APICALL
CompilerGetProperties(gc_compiler_handle_t compiler, gc_compiler_properties_t *compiler_info);

GC_DLLEXPORT gc_result_t GC_APICALL
ExecutableCreate(gc_compiler_handle_t compiler, gc_executable_desc_t desc, gc_executable_handle_t *executable);

GC_DLLEXPORT gc_result_t GC_APICALL
ExecutableGetSerializableBlob(gc_executable_handle_t executable, uint32_t *blob_size, const uint8_t **blob);

GC_DLLEXPORT gc_result_t GC_APICALL
ExecutableGetBufferProperties(gc_executable_handle_t executable, uint32_t *buffer_count, gc_buffer_properties_t *buffers);

GC_DLLEXPORT gc_result_t GC_APICALL
ExecutableInitialize(gc_executable_handle_t executable, gc_initialize_schedule_desc_t desc);

GC_DLLEXPORT gc_result_t GC_APICALL
ExecutableDestroy(gc_executable_handle_t executable);

GC_DLLEXPORT gc_result_t GC_APICALL
CompilerDestroy(gc_compiler_handle_t compiler);

#if defined(__cplusplus)
} // extern "C"
#endif

// clang-format on
#endif  // GRAPH_COMPILER_API_H
