// {% copyright %}
///
/// @file
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     FP16<->U8 quantize/dequantize util
///

#pragma once

#include <mv_types.h>

// Dequantize
void uint8ToFp16_EqualScales(const uchar8 *u8In, uint32_t inBytes, half8 *fpOut, half scale, half zero);

void uint8ToFp16_PerChannelScales(const uchar8 *u8In, uint32_t inBytes, half8 *fpOut, const half *scales, uint16_t numScales, half zero);

// Quantize
void fp16ToUint8_EqualScales(const half8 *fpIn, uint32_t inBytes, uchar8 *u8Out, half scale, half zero);

