///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      points_penta.h
/// 

#include "osdDefs.h"

//convex pentagon

OsdPoint vertsPenta[] ALIGNED(16) = {
    { 2 *  63.0f,  2 *   0.0f },
    { 2 * 127.0f,  2 *  41.0f },
    { 2 *  96.0f,  2 * 127.0f },
    { 2 *  14.0f,  2 * 111.0f },
    { 2 *   0.5f,  2*   60.0f }
};