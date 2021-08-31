// {% copyright %}
///
/// @file
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     FP16<->U8 quantize/dequantize util
///

#include <assert.h>
#include <moviVectorUtils.h>

#include "quant_util.h"

/// real_value = scale * (quantized_value - zero_point)
void uint8ToFp16_EqualScales(const uchar8 *u8In, uint32_t inBytes, half8 *fpOut, half scale, half zero) {
    const uint32_t vecBytes = inBytes - (inBytes % 8);

    const half8 zeros = { zero, zero, zero, zero, zero, zero, zero, zero };
    const half8 scales = { scale, scale, scale, scale, scale, scale, scale, scale };
    const uint8_t *end = reinterpret_cast<const uint8_t *>(u8In) + vecBytes;

    uint8_t *u8Step = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(u8In));
    uint8_t *fpStep = reinterpret_cast<uint8_t *>(fpOut);
    half8 val;

    do {
        val = __builtin_shave_lsu_ld128_u8_f16_r(reinterpret_cast<uchar8 *>(u8Step));
        // val = val - zero;
        // val = val * scale;
        val = __builtin_shave_vau_sub_f16_rr(val, zeros);  /* VAU.SUB.f16 */
        val = __builtin_shave_vau_mul_f16_rr(val, scales); /* VAU.MUL.f16 */

        *reinterpret_cast<half8 *>(fpStep) = val;
        u8Step = (uint8_t *)(u8Step + sizeof(uchar8));
        fpStep = (uint8_t *)(fpStep + sizeof(half8));
    } while (u8Step < end);

    // Finish up the channels that didn't fill a full vector (1-7 iterations)
    uint8_t *u8Tail = (uint8_t *)u8Step;
    half *fpTail = (half *)fpStep;

    for (uint32_t i = 0; i < inBytes - vecBytes; i++) {
        fpTail[i] = scale * ((half)u8Tail[i] - zero);
    }
}

/// real_value = scale * (quantized_value - zero_point)
void uint8ToFp16_PerChannelScales(const uchar8 *u8In, uint32_t inBytes, half8 *fpOut, const half *scalesPerChannel,
                                  uint16_t numScales, half zero) {
    assert(inBytes % numScales == 0);

    const half8 zeros = { zero, zero, zero, zero, zero, zero, zero, zero };

    uint8_t *u8Step = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(u8In));
    uint8_t *fpStep = reinterpret_cast<uint8_t *>(fpOut);
    half8 val;

    // Have two competing problems: filling the vector unit and returning with all scales consumed for the next
    // invocation Choose to return with all scales/channels consumed
    const half *startScales, *endScales;
    half8 scales = { 0, 0, 0, 0, 0, 0, 0, 0 };

    // Divide round down by numScales and then 8 channel vector
    const unsigned int numBlocks = ((inBytes / numScales) * numScales) / 8;

    unsigned int curBlock = 0;
    for (unsigned int i = 0; i < numBlocks; i++) {
        // Calculate scales ptr start, end. If start < end, then do one read. Else do it piecewise
        startScales = &scalesPerChannel[curBlock % numScales];
        endScales = &scalesPerChannel[(curBlock + 7) % numScales];

        if (numScales > 7 && startScales < endScales) {
            // Scales can be read in as an 8 element vector
            scales = *((half8 *)startScales);
        } else {
            scales[0] = *startScales;
            scales[1] = scalesPerChannel[(curBlock + 1) % numScales];
            scales[2] = scalesPerChannel[(curBlock + 2) % numScales];
            scales[3] = scalesPerChannel[(curBlock + 3) % numScales];
            scales[4] = scalesPerChannel[(curBlock + 4) % numScales];
            scales[5] = scalesPerChannel[(curBlock + 5) % numScales];
            scales[6] = scalesPerChannel[(curBlock + 6) % numScales];
            scales[7] = *endScales;
        }

        val = __builtin_shave_lsu_ld128_u8_f16_r(reinterpret_cast<uchar8 *>(u8Step));
        // val = val - zero;
        // val = val * scale;
        val = __builtin_shave_vau_sub_f16_rr(val, zeros);  /* VAU.SUB.f16 */
        val = __builtin_shave_vau_mul_f16_rr(val, scales); /* VAU.MUL.f16 */

        *reinterpret_cast<half8 *>(fpStep) = val;
        curBlock += 8;
        u8Step = (uint8_t *)(u8Step + sizeof(uchar8));
        fpStep = (uint8_t *)(fpStep + sizeof(half8));
        inBytes -= 8;
    }

    // Finish up the channels that didn't fill a full vector (1-7 iterations)
    uint8_t *u8Tail = (uint8_t *)u8Step;
    half *fpTail = (half *)fpStep;
    for (uint32_t i = 0, j = curBlock % numScales; i < inBytes; i++, j++) {
        fpTail[i] = scalesPerChannel[j % numScales] * ((half)u8Tail[i] - zero);
    }
}

/// quantized_value = zero_point + real_value / scale
void fp16ToUint8_EqualScales(const half8 *fpIn, uint32_t inBytes, uchar8 *u8Out, half scale, half zero) {
    // Process in chunks of 16 bytes (8 halfs == 128 bit vector)
    const uint32_t vecBytes = inBytes - (inBytes % 16);

    const uint8_t *end = reinterpret_cast<const uint8_t *>(fpIn) + vecBytes;
    uint8_t *fpStep = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(fpIn));
    uint8_t *u8Step = reinterpret_cast<uint8_t *>(u8Out);
    half8 val;

    do {
        val = *reinterpret_cast<half8 *>(fpStep);
        val = val / scale;
        val = val + zero + 0.5;

        __builtin_shave_lsu_st128_f16_u8_rr(val, reinterpret_cast<uchar8 *>(u8Step)); /* LSU.ST128.f16.u8 */

        u8Step = (uint8_t *)(u8Step + sizeof(uchar8));
        fpStep = (uint8_t *)(fpStep + sizeof(half8));
    } while (fpStep < end);

    // Finish up the channels that didn't fill a full vector (1-7 iterations)
    uint8_t *u8Tail = (uint8_t *)u8Step;
    half *fpTail = (half *)fpStep;
    uint32_t tailElems = (inBytes - vecBytes) / 2;

    for (uint32_t i = 0; i < tailElems; i++) {
        u8Tail[i] = (uchar)(zero + fpTail[i] / scale);
    }
}
