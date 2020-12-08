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
/// @file      points_line.h
/// 

#include "osdDefs.h"

#define LN_W  400 //line width

OsdPoint vertsThick1[] ALIGNED(16) = {
  #define THK 1.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
  #undef THK
};

OsdPoint vertsThick2[] ALIGNED(16) = {
#define THK 2.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
#undef THK
};

OsdPoint vertsThick3[] ALIGNED(16) = {
#define THK 3.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
#undef THK
};
OsdPoint vertsThick5[] ALIGNED(16) = {
#define THK 5.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
#undef THK
};
#undef LN_W