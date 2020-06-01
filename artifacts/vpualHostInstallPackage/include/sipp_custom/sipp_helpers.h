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
/// @file      sipp_filter_helpers.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for Host side SIPP helpers.
///

#ifndef _SIPP_FILTER_HELPERS_H_
#define _SIPP_FILTER_HELPERS_H_

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Converts a floating point number to it's integer representation for floating point 16 as most filters need.
 *
 * @param x - floating point number to convert
 */
uint16_t fp32tofp16( float x );

#ifdef __cplusplus
}
#endif

#endif /* _SIPP_FILTER_HELPERS_H_ */
