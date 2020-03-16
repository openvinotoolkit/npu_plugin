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
