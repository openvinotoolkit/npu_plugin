#ifndef CUSTOM_MATH_HPP
#define CUSTOM_MATH_HPP

#include <stdint.h>
#include <utility>
#include <vector>
#include "include/mcm/tensor/dtype/dtype.hpp"


namespace mv
{
    unsigned round_up(unsigned x, unsigned mult);
    unsigned ceil_division(unsigned x, unsigned d);
    unsigned count_bits(unsigned number);
    unsigned next_greater_power_of_2(unsigned number);
    std::vector<std::size_t> tileSpatialOutputSize(std::size_t outputSize , std::size_t numberOfSplits);
    uint16_t getWindowSize(uint16_t kx, uint16_t sx, mv::DType dataType);


    // FP16 Conversion
    union bit_field32
    {
        float fp;
        int si;
        unsigned int ui;
    };

    static int const shiftFraction = 13;    // difference in size of fraction field
    static int const shiftSign = 16;        // difference in size of number representation

    static int const infN  = 0x7F800000;    // flt32 infinity
    static int const inf16 = 0x00007C00;    // flt16 infinity
    static int const maxNorm  = 0x477FE000; // max flt16 normal as a flt32 = 2**(15)*1.11111
    static int const minNorm  = 0x38800000; // min flt16 normal as a flt32 = 2**(-14)*1.0000
    static int const sign32= 0x80000000;    // flt32 sign bit
    static unsigned int const roundBit = 0x00001000; // highest order truncated fraction bit

    static int const NaNC = 0x7FFF;         // max unsigned fp16 Not a Number
    static int const infC = infN >> shiftFraction;
    static int const nanN = (infC + 1) << shiftFraction; // minimum fp16 nan as fp32 (w/noise)
    static int const maxC = maxNorm >> shiftFraction;
    static int const minC = minNorm >> shiftFraction;
    static int const signC = sign32 >> shiftSign; // flt16 sign bit

    static int const mulN = 0x52000000; // (1 << 23) / minNorm
    static int const mulC = 0x33800000; // minNorm / (1 << (23 - shift))

    static int const subC = 0x003FF; // max flt32 subnormal down shifted
    static int const norC = 0x00400; // min flt32 normal down shifted

    static int const maxD = infC - maxC - 1;
    static int const minD = minC - subC - 1;

    uint16_t fp32_to_fp16(float value);
    uint16_t fp32_to_fp16(double value);
    float fp16_to_fp32(uint16_t value);
}

#endif
