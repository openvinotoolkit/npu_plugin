/**************************************************************************//**
*
* INTEL CONFIDENTIAL
* Copyright 2021
* Intel Corporation All Rights Reserved.
*
* The source code contained or described herein and all documents related to the
* source code ("Material") are owned by Intel Corporation or its suppliers or
* licensors. Title to the Material remains with Intel Corporation or its suppliers
* and licensors. The Material contains trade secrets and proprietary and confidential
* information of Intel or its suppliers and licensors. The Material is protected by
* worldwide copyright and trade secret laws and treaty provisions. No part of the
* Material may be used, copied, reproduced, modified, published, uploaded, posted
* transmitted, distributed, or disclosed in any way without Intel's prior express
* written permission.
*
* No license under any patent, copyright, trade secret or other intellectual
* property right is granted to or conferred upon you by disclosure or delivery
* of the Materials, either expressly, by implication, inducement, estoppel
* or otherwise. Any license under such intellectual property rights must be
* express and approved by Intel in writing.
*
* @file ze_graph_ext.h
*
******************************************************************************/
#pragma once

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphCreate_ext_t)(
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    const ze_graph_desc_t* desc,                    ///< [in] pointer to graph descriptor
    ze_graph_handle_t* phGraph                      ///< [out] pointer to handle of graph object created
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphDestroy_ext_t)(
    ze_graph_handle_t hGraph                        ///< [in][release] handle of graph object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetProperties_ext_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    ze_graph_properties_t* pGraphProperties         ///< [in,out] query result for graph properties.
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetArgumentProperties_ext_t)(
    ze_graph_handle_t hGraph,                                       ///< [in] handle of the graph object
    uint32_t argIndex,                                              ///< [in] index of the argument to get properties
    ze_graph_argument_properties_t* pGraphArgumentProperties        ///< [in,out] query result for graph argument properties.
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphSetArgumentValue_ext_t)(
    ze_graph_handle_t hGraph,
    uint32_t argIndex,
    const void* pArgValue
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnAppendGraphInitialize_ext_t)(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_graph_handle_t hGraph                        ///< [in] handle of the graph
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnAppendGraphExecute_ext_t)(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_graph_handle_t hGraph                        ///< [in] handle of the graph
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Graph functions pointers
typedef struct _ze_graph_dditable_ext_t
{
    ze_pfnGraphCreate_ext_t                                     pfnCreate;
    ze_pfnGraphDestroy_ext_t                                    pfnDestroy;
    ze_pfnGraphGetProperties_ext_t                              pfnGetProperties;
    ze_pfnGraphGetArgumentProperties_ext_t                      pfnGetArgumentProperties; 
    ze_pfnGraphSetArgumentValue_ext_t                           pfnSetArgumentValue;
    ze_pfnAppendGraphInitialize_ext_t                           pfnAppendGraphInitialize;
    ze_pfnAppendGraphExecute_ext_t                              pfnAppendGraphExecute;
} ze_graph_dditable_ext_t;