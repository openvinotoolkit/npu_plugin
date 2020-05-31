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
/// @file      points_heart.h
/// 

#include "osdDefs.h"

#define MID 179.0f   //on X-axis

OsdPoint vertsHeart[] ALIGNED(16) = {
 //[Inflection point]
    { MID +   0.0f,  91.0f},

 //[Right lobe]
    { MID +   1.0f,  79.0f},
    { MID +   2.0f,  70.0f},
    { MID +   4.0f,  63.0f},
    { MID +   6.0f,  56.0f},
    { MID +   9.0f,  49.0f},
    { MID +  14.0f,  41.0f},
    { MID +  19.0f,  34.0f},
    { MID +  24.0f,  28.0f},
    { MID +  29.0f,  23.0f},
    { MID +  34.0f,  19.0f},
    { MID +  39.0f,  15.0f},
    { MID +  44.0f,  12.0f},
    { MID +  50.0f,   9.0f},
    { MID +  56.0f,   6.0f},
    { MID +  62.0f,   4.0f},
    { MID +  69.0f,   2.0f},
    { MID +  82.0f,   0.0f},
    { MID +  95.0f,   0.0f},
    { MID + 103.0f,   1.0f},
    { MID + 110.0f,   2.0f},
    { MID + 124.0f,   7.0f},
    { MID + 130.0f,  10.0f},
    { MID + 136.0f,  13.0f},
    { MID + 142.0f,  17.0f},
    { MID + 148.0f,  22.0f},
    { MID + 154.0f,  28.0f},
    { MID + 161.0f,  36.0f},
    { MID + 167.0f,  45.0f},
    { MID + 171.0f,  54.0f},
    { MID + 175.0f,  65.0f},
    { MID + 178.0f,  76.0f},
    { MID + 179.0f,  91.0f},
    { MID + 178.0f, 106.0f},
    { MID + 177.0f, 118.0f},
    { MID + 173.0f, 138.0f},
    { MID + 169.0f, 153.0f},
    { MID + 163.0f, 170.0f},
    { MID + 156.0f, 185.0f},
    { MID + 148.0f, 200.0f},
    { MID + 138.0f, 215.0f},
    { MID + 129.0f, 227.0f},
    { MID + 119.0f, 239.0f},
    { MID +  93.0f, 265.0f},

   //[Bottom point]
    { MID +   0.0f, 358.0f},

   //[Left lobe] simmetrical
    { MID -  93.0f, 265.0f},
    { MID - 119.0f, 239.0f},
    { MID - 129.0f, 227.0f},
    { MID - 138.0f, 215.0f},
    { MID - 148.0f, 200.0f},
    { MID - 156.0f, 185.0f},
    { MID - 163.0f, 170.0f},
    { MID - 169.0f, 153.0f},
    { MID - 173.0f, 138.0f},
    { MID - 177.0f, 118.0f},
    { MID - 178.0f, 106.0f},
    { MID - 179.0f,  91.0f},
    { MID - 178.0f,  76.0f},
    { MID - 175.0f,  65.0f},
    { MID - 171.0f,  54.0f},
    { MID - 167.0f,  45.0f},
    { MID - 161.0f,  36.0f},
    { MID - 154.0f,  28.0f},
    { MID - 148.0f,  22.0f},
    { MID - 142.0f,  17.0f},
    { MID - 136.0f,  13.0f},
    { MID - 130.0f,  10.0f},
    { MID - 124.0f,   7.0f},
    { MID - 110.0f,   2.0f},
    { MID - 103.0f,   1.0f},
    { MID -  95.0f,   0.0f},
    { MID -  82.0f,   0.0f},
    { MID -  69.0f,   2.0f},
    { MID -  62.0f,   4.0f},
    { MID -  56.0f,   6.0f},
    { MID -  50.0f,   9.0f},
    { MID -  44.0f,  12.0f},
    { MID -  39.0f,  15.0f},
    { MID -  34.0f,  19.0f},
    { MID -  29.0f,  23.0f},
    { MID -  24.0f,  28.0f},
    { MID -  19.0f,  34.0f},
    { MID -  14.0f,  41.0f},
    { MID -   9.0f,  49.0f},
    { MID -   6.0f,  56.0f},
    { MID -   4.0f,  63.0f},
    { MID -   2.0f,  70.0f},
    { MID -   1.0f,  79.0f},
};

#undef MID