/*
 *
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#pragma once

#include "ze_api.h"

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's graph object
typedef struct _ze_graph_handle_t* ze_graph_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported graph creation input formats
typedef enum _ze_graph_format_t {
    ZE_GRAPH_FORMAT_IL_MCM = 0,  ///< Format is MCM IL format
    ZE_GRAPH_FORMAT_NATIVE,      ///< Format is device native format

} ze_graph_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph descriptor
typedef struct _ze_graph_desc_t {
    ze_graph_format_t format;    ///< [in] Graph format passed in with pInputGraph
    size_t inputSize;            ///< [in] Size of graph in bytes
    const uint8_t* pInputGraph;  ///< [in] Pointer to graph IL or native binary
} ze_graph_desc_t;

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeGraphCreate(
        ze_device_handle_t hDevice,   ///< [in] handle of the device
        const ze_graph_desc_t* desc,  ///< [in] pointer to graph descriptor
        ze_graph_handle_t* phGraph    ///< [out] pointer to handle of graph object created
);

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeGraphDestroy(
        ze_graph_handle_t hGraph  ///< [in][release] handle of graph object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph properties
typedef struct _ze_graph_properties_t {
    uint32_t numGraphArgs;  ///< [out] number of graph arguments

} ze_graph_properties_t;

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeGraphGetProperties(
        ze_graph_handle_t hGraph,                ///< [in] handle of the graph object
        ze_graph_properties_t* pGraphProperties  ///< [in,out] query result for graph properties.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_type_t {
    ZE_GRAPH_ARGUMENT_TYPE_INPUT,
    ZE_GRAPH_ARGUMENT_TYPE_OUTPUT,

} ze_graph_argument_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_dimension_t {
    ZE_GRAPH_ARGUMENT_DIMENSION_N,
    ZE_GRAPH_ARGUMENT_DIMENSION_C,
    ZE_GRAPH_ARGUMENT_DIMENSION_H,
    ZE_GRAPH_ARGUMENT_DIMENSION_W,
    ZE_GRAPH_ARGUMENT_DIMENSION_D,

    ZE_GRAPH_ARGUMENT_DIMENSION_MAX,

} ze_graph_argument_dimension_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_precision_t {
    ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN,
    ZE_GRAPH_ARGUMENT_PRECISION_FP32,
    ZE_GRAPH_ARGUMENT_PRECISION_FP16,
    ZE_GRAPH_ARGUMENT_PRECISION_UINT16,
    ZE_GRAPH_ARGUMENT_PRECISION_UINT8,
    ZE_GRAPH_ARGUMENT_PRECISION_INT32,
    ZE_GRAPH_ARGUMENT_PRECISION_INT16,
    ZE_GRAPH_ARGUMENT_PRECISION_INT8,
    ZE_GRAPH_ARGUMENT_PRECISION_BIN,

} ze_graph_argument_precision_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_layout_t {
    ZE_GRAPH_ARGUMENT_LAYOUT_ANY = 0x00,

    ZE_GRAPH_ARGUMENT_LAYOUT_NCHW,
    ZE_GRAPH_ARGUMENT_LAYOUT_NHWC,
    ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW,
    ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC,

    ZE_GRAPH_ARGUMENT_LAYOUT_OIHW = 0x40,

    ZE_GRAPH_ARGUMENT_LAYOUT_C = 0x60,

    ZE_GRAPH_ARGUMENT_LAYOUT_CHW = 0x80,

    ZE_GRAPH_ARGUMENT_LAYOUT_HW = 0xC0,
    ZE_GRAPH_ARGUMENT_LAYOUT_NC,
    ZE_GRAPH_ARGUMENT_LAYOUT_CN,

    ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED = 0xC8

} ze_graph_argument_layout_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_ARGUMENT_NAME
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_ARGUMENT_NAME 256
#endif  // ZE_MAX_GRAPH_ARGUMENT_NAME

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE ZE_GRAPH_ARGUMENT_DIMENSION_MAX
#endif  // ZE_MAX_GRAPH_ARGUMENT_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef struct _ze_graph_argument_properties_t {
    char name[ZE_MAX_GRAPH_ARGUMENT_NAME];  ///< [out] Graph argument name
    ze_graph_argument_type_t type;
    uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
    ze_graph_argument_precision_t precision;
    ze_graph_argument_layout_t layout;

} ze_graph_argument_properties_t;

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeGraphGetArgumentProperties(
        ze_graph_handle_t hGraph,  ///< [in] handle of the graph object
        uint32_t argIndex,         ///< [in] index of the argument to get properties
        ze_graph_argument_properties_t*
                pGraphArgumentProperties  ///< [in,out] query result for graph argument properties.
);

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeGraphSetArgumentValue(ze_graph_handle_t hGraph, uint32_t argIndex,
                                                            const void* pArgValue);

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeAppendGraphInitialize(
        ze_command_list_handle_t hCommandList,  ///< [in] handle of the command list
        ze_graph_handle_t hGraph                ///< [in] handle of the graph
);

///////////////////////////////////////////////////////////////////////////////
ZE_APIEXPORT ze_result_t ZE_APICALL zeAppendGraphExecute(
        ze_command_list_handle_t hCommandList,  ///< [in] handle of the command list
        ze_graph_handle_t hGraph                ///< [in] handle of the graph
);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnGraphCreate_ext_t)(
        ze_device_handle_t hDevice,   ///< [in] handle of the device
        const ze_graph_desc_t* desc,  ///< [in] pointer to graph descriptor
        ze_graph_handle_t* phGraph    ///< [out] pointer to handle of graph object created
);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnGraphDestroy_ext_t)(
        ze_graph_handle_t hGraph  ///< [in][release] handle of graph object to destroy
);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnGraphGetProperties_ext_t)(
        ze_graph_handle_t hGraph,                ///< [in] handle of the graph object
        ze_graph_properties_t* pGraphProperties  ///< [in,out] query result for graph properties.
);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnGraphGetArgumentProperties_ext_t)(
        ze_graph_handle_t hGraph,  ///< [in] handle of the graph object
        uint32_t argIndex,         ///< [in] index of the argument to get properties
        ze_graph_argument_properties_t*
                pGraphArgumentProperties  ///< [in,out] query result for graph argument properties.
);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnGraphSetArgumentValue_ext_t)(ze_graph_handle_t hGraph, uint32_t argIndex,
                                                                   const void* pArgValue);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnAppendGraphInitialize_ext_t)(
        ze_command_list_handle_t hCommandList,  ///< [in] handle of the command list
        ze_graph_handle_t hGraph                ///< [in] handle of the graph
);

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t(ZE_APICALL* ze_pfnAppendGraphExecute_ext_t)(
        ze_command_list_handle_t hCommandList,  ///< [in] handle of the command list
        ze_graph_handle_t hGraph                ///< [in] handle of the graph
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Graph functions pointers
typedef struct _ze_graph_dditable_ext_t {
    ze_pfnGraphCreate_ext_t pfnCreate;
    ze_pfnGraphDestroy_ext_t pfnDestroy;
    ze_pfnGraphGetProperties_ext_t pfnGetProperties;
    ze_pfnGraphGetArgumentProperties_ext_t pfnGetArgumentProperties;
    ze_pfnGraphSetArgumentValue_ext_t pfnSetArgumentValue;
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize;
    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute;
} ze_graph_dditable_ext_t;
