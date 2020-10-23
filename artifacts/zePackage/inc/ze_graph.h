/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ze_graph.h
 *
 * @brief Intel 'One API' Level-Zero APIs for Graph
 *
 */
 
#ifndef _ZE_GRAPH_H
#define _ZE_GRAPH_H
#if defined(__cplusplus)
#pragma once
#endif
#if !defined(_ZE_API_H)
#pragma message("warning: this file is not intended to be included directly")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief API version of ::ze_graph_desc_t
typedef enum _ze_graph_desc_version_t
{
    ZE_GRAPH_DESC_VERSION_CURRENT = ZE_MAKE_VERSION(0, 91),  ///< version 0.91

} ze_graph_desc_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported graph creation input formats
typedef enum _ze_graph_format_t
{
    ZE_GRAPH_FORMAT_IL_MCM = 0,                    ///< Format is MCM IL format
    ZE_GRAPH_FORMAT_NATIVE,                        ///< Format is device native format

} ze_graph_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph descriptor
typedef struct _ze_graph_desc_t
{
    ze_graph_desc_version_t version;                ///< [in] ::ZE_GRAPH_DESC_VERSION_CURRENT
    ze_graph_format_t format;                       ///< [in] Graph format passed in with pInputGraph
    size_t inputSize;                               ///< [in] Size of graph in bytes
    const uint8_t* pInputGraph;                     ///< [in] Pointer to graph IL or native binary
} ze_graph_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates graph object from an input IL or native binary.
/// 
/// @details
///     - Compiles the graph for execution on the device.
///     - The graph can only be used on the device on which it was created.
///     - The graph can be copied to other devices within the same driver
///       instance by using ::zeGraphGetNativeBinary.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == desc->pInputGraph`
///         + `nullptr == phGraph`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
///         + `::ZE_GRAPH_DESC_VERSION_CURRENT < desc->version`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + desc->format
///     - ::ZE_RESULT_ERROR_INVALID_NATIVE_BINARY
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `0 == desc->inputSize`
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_MODULE_BUILD_FAILURE
__ze_api_export ze_result_t __zecall
zeGraphCreate(
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    const ze_graph_desc_t* desc,                    ///< [in] pointer to graph descriptor
    ze_graph_handle_t* phGraph                      ///< [out] pointer to handle of graph object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys graph
/// 
/// @details
///     - The application is responsible for making sure the device is not
///       currently referencing the graph before it is deleted
///     - The implementation of this function will immediately free all Host and
///       Device allocations associated with this graph
///     - The application may **not** call this function from simultaneous
///       threads with the same graph handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hGraph`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
__ze_api_export ze_result_t __zecall
zeGraphDestroy(
    ze_graph_handle_t hGraph                        ///< [in][release] handle of the graph object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve native binary from Graph.
/// 
/// @details
///     - The native binary output can be cached to disk and new graphs can be
///       later constructed from the cached copy.
///     - The native binary will retain debugging information that is associated
///       with a graph.
///     - The caller can pass nullptr for pGraphNativeBinary when querying only
///       for size.
///     - The implementation will copy the native binary into a buffer supplied
///       by the caller.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == pGraph`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
__ze_api_export ze_result_t __zecall
zeGraphGetNativeBinary(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph
    size_t* pSize,                                  ///< [in,out] size of native binary in bytes.
    uint8_t* pGraphNativeBinary                     ///< [in,out][optional] byte pointer to native binary
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief API version of ::ze_graph_properties_t
typedef enum _ze_graph_properties_version_t
{
    ZE_GRAPH_PROPERTIES_VERSION_CURRENT = ZE_MAKE_VERSION(0, 91),///< version 0.91

} ze_graph_properties_version_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_NAME
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_NAME  256
#endif // ZE_MAX_GRAPH_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph properties
typedef struct _ze_graph_properties_t
{
    ze_graph_properties_version_t version;          ///< [in] ::ZE_GRAPH_PROPERTIES_VERSION_CURRENT
    uint32_t numGraphArgs;                          ///< [out] number of graph arguments

} ze_graph_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve graph properties.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hGraph`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pGraphProperties`
__ze_api_export ze_result_t __zecall
zeGraphGetProperties(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    ze_graph_properties_t* pGraphProperties         ///< [in,out] query result for graph properties.
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_ARGUMENT_NAME
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_ARGUMENT_NAME  256
#endif // ZE_MAX_GRAPH_ARGUMENT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief API version of ::ze_graph_argument_properties_version_t
typedef enum _ze_graph_argument_properties_version_t
{
    ZE_GRAPH_ARGUMENT_PROPERTIES_VERSION_CURRENT = ZE_MAKE_VERSION(0, 91),///< version 0.91

} ze_graph_argument_properties_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_type_t
{
    ZE_GRAPH_ARGUMENT_TYPE_INPUT,                   ///< version 0.91
    ZE_GRAPH_ARGUMENT_TYPE_OUTPUT,                  ///< version 0.91

} ze_graph_argument_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_dimension_t
{
    ZE_GRAPH_ARGUMENT_DIMENSION_N,
    ZE_GRAPH_ARGUMENT_DIMENSION_C,
    ZE_GRAPH_ARGUMENT_DIMENSION_H,
    ZE_GRAPH_ARGUMENT_DIMENSION_W,
    ZE_GRAPH_ARGUMENT_DIMENSION_D,

    ZE_GRAPH_ARGUMENT_DIMENSION_MAX,

} ze_graph_argument_dimension_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_precision_t
{
    ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN,            ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_FP32,               ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_FP16,               ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_UINT16,             ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_UINT8,              ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_INT32,              ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_INT16,              ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_INT8,               ///< version 0.91
    ZE_GRAPH_ARGUMENT_PRECISION_BIN,                ///< version 0.91

} ze_graph_argument_precision_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_layout_t
{
    ZE_GRAPH_ARGUMENT_LAYOUT_ANY        = 0x00,     ///< version 0.91

    ZE_GRAPH_ARGUMENT_LAYOUT_NCHW,                  ///< version 0.91
    ZE_GRAPH_ARGUMENT_LAYOUT_NHWC,                  ///< version 0.91
    ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW,                 ///< version 0.91
    ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC,                 ///< version 0.91

    ZE_GRAPH_ARGUMENT_LAYOUT_OIHW       = 0x40,     ///< version 0.91

    ZE_GRAPH_ARGUMENT_LAYOUT_C          = 0x60,     ///< version 0.91

    ZE_GRAPH_ARGUMENT_LAYOUT_CHW        = 0x80,     ///< version 0.91

    ZE_GRAPH_ARGUMENT_LAYOUT_HW         = 0xC0,     ///< version 0.91
    ZE_GRAPH_ARGUMENT_LAYOUT_NC,                    ///< version 0.91
    ZE_GRAPH_ARGUMENT_LAYOUT_CN,                    ///< version 0.91

    ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED    = 0xC8      ///< version 0.91

} ze_graph_argument_layout_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE  ZE_GRAPH_ARGUMENT_DIMENSION_MAX
#endif // ZE_MAX_GRAPH_ARGUMENT_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef struct _ze_graph_argument_properties_t
{
    ze_graph_argument_properties_version_t version;     ///< [in] ::ZE_GRAPH_ARGUMENT_PROPERTIES_VERSION_CURRENT

    char name[ZE_MAX_GRAPH_ARGUMENT_NAME];              ///< [out] Graph argument name
    ze_graph_argument_type_t type;
    uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
    ze_graph_argument_precision_t precision;
    ze_graph_argument_layout_t layout;

} ze_graph_argument_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve graph argument properties.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hGraph`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pGraphArgumentProperties`
__ze_api_export ze_result_t __zecall
zeGraphGetArgumentProperties(
    ze_graph_handle_t hGraph,                                   ///< [in] handle of the graph object
    uint32_t argIndex,                                          ///< [in] index of the argument to get properties
    ze_graph_argument_properties_t* pGraphArgumentProperties    ///< [in,out] query result for graph argument properties.
);

///////////////////////////////////////////////////////////////////////////////
__ze_api_export ze_result_t __zecall
zeGraphSetArgumentValue(
    ze_graph_handle_t hGraph,                                   ///< [in] handle of the graph object
    uint32_t argIndex,                                          ///< [in] index of the argument to get properties
    const void* pArgValue                                       ///< [in] pointer to argument value
);

///////////////////////////////////////////////////////////////////////////////
__ze_api_export ze_result_t __zecall
zeCommandListAppendGraphInitialize(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_graph_handle_t hGraph                        ///< [in] handle of the graph
    );

///////////////////////////////////////////////////////////////////////////////
__ze_api_export ze_result_t __zecall
zeCommandListAppendGraphExecute(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_graph_handle_t hGraph                        ///< [in] handle of the graph
);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZE_GRAPH_H
