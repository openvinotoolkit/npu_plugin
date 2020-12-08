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
/// @file      points_small.h
/// 

#include "osdDefs.h"

//Small triangle
OsdPoint vertsSmall[] ALIGNED(16) = {
  #define TSZ 4.0f
    { 90.0f      , 70.0f },
    { 90.0f+TSZ  , 70.0f + TSZ/2 },
    { 90.0f+TSZ/2, 70.0f + TSZ },
  #undef  TSZ
};