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
* @file ze_fence_ext.h
*
******************************************************************************/
#pragma once

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceDeviceSignal_ext
typedef ze_result_t (ZE_APICALL *ze_pfnFenceDeviceSignal_ext_t)(
    ze_fence_handle_t,
    uint64_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceDeviceSynchronize_ext
typedef ze_result_t (ZE_APICALL *ze_pfnFenceDeviceSynchronize_ext_t)(
    ze_command_queue_handle_t,
    ze_fence_handle_t,
    uint64_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Fence extension functions pointers
typedef struct _ze_fence_dditable_ext_t
{
    ze_pfnFenceDeviceSignal_ext_t         pfnDeviceSignal;
    ze_pfnFenceDeviceSynchronize_ext_t    pfnDeviceSynchronize;
} ze_fence_dditable_ext_t;