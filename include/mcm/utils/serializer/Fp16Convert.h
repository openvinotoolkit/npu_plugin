///
/// @file
/// @copyright All code copyright Movidius Ltd 2014, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @defgroup Fp16Convert Fp16 Convert
/// @{
/// @brief Fp16 manipulation and conversion utility
///        minimal set of fp16 conversions functions for
///        sharing data between Leon and SHAVES or other HW blocks
///        which expect fp16 data

#ifndef __FP16_CONVERT_H__
#define __FP16_CONVERT_H__

#include "include/mcm/utils/serializer/mv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOVIDIUS_FP32

/// @name
/// Rounding modes
/// @{

#define F32_RND_NEAREST_EVEN     0
#define F32_RND_MINUS_INF        1
#define F32_RND_PLUS_INF         2
#define F32_RND_TO_ZERO          3
/// @}

/// @name
/// Detect tinyness mode
/// @{

#define F32_DETECT_TINY_AFTER_RND  0
#define F32_DETECT_TINY_BEFORE_RND 1
/// @}

/// @name
/// Exceptions
/// @{

#define F32_EX_INEXACT     0x00000001//0x00000020
#define F32_EX_DIV_BY_ZERO 0x00000002//0x00000004
#define F32_EX_INVALID     0x00000004//0x00000001
#define F32_EX_UNDERFLOW   0x00000008//0x00000010
#define F32_EX_OVERFLOW    0x00000010//0x00000008
/// @}

#define F32_NAN_DEFAULT    0xFFC00000

// Macros
#define EXTRACT_F16_SIGN(x)   ((x >> 15) & 0x1)
#define EXTRACT_F16_EXP(x)    ((x >> 10) & 0x1F)
#define EXTRACT_F16_FRAC(x)   (x & 0x000003FF)
#define EXTRACT_F32_SIGN(x)   ((x >> 31) & 0x1)
#define EXTRACT_F32_EXP(x)    ((x >> 23) & 0xFF)
#define EXTRACT_F32_FRAC(x)   (x & 0x007FFFFF)
#define RESET_SNAN_BIT(x)     x = x | 0x00400000

#define PACK_F32(x, y, z)     ((x << 31) + (y << 23) + z)
#define PACK_F16(x, y, z)     ((x << 15) + (y << 10) + z)

#define F16_IS_NAN(x)       ((x & 0x7FFF)> 0x7C00)
#define F16_IS_SNAN(x)      (((x & 0x7E00) == 0x7C00)&&((x & 0x1FF)> 0))
#define F32_IS_NAN(x)       ((x & 0x7FFFFFFF)> 0x7F800000)
#define F32_IS_SNAN(x)      (((x & 0x7FC00000) == 0x7F800000)&&((x & 0x3FFFFF)> 0))

extern unsigned int rnd_mode;
extern unsigned int exceptionsReg;
extern unsigned int* exceptions;

unsigned int f16_shift_left(unsigned int op, unsigned int cnt);

class Float16Compressor
    {
        union Bits
        {
            float f;
            int32_t si;
            uint32_t ui;
        };

        static int const shift = 13;
        static int const shiftSign = 16;

        static int32_t const infN = 0x7F800000; // flt32 infinity
        static int32_t const maxN = 0x477FE000; // max flt16 normal as a flt32
        static int32_t const minN = 0x38800000; // min flt16 normal as a flt32
        static int32_t const signN = 0x80000000; // flt32 sign bit
        static uint32_t const roundBit = 0x00001000; // highest order truncated fraction bit
        static int32_t const NaNC = 0x7FFF;         // fp16 Not a Number

        static int32_t const infC = infN >> shift;
        static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
        static int32_t const maxC = maxN >> shift;
        static int32_t const minC = minN >> shift;
        static int32_t const signC = signN >> shiftSign; // flt16 sign bit

        static int32_t const mulN = 0x52000000; // (1 << 23) / minN
        static int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

        static int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
        static int32_t const norC = 0x00400; // min flt32 normal down shifted

        static int32_t const maxD = infC - maxC - 1;
        static int32_t const minD = minC - subC - 1;
    public:

        static uint16_t compress(float value)
        {
            Bits v, s;
            v.f = value;

            uint32_t sign = v.si & signN;   // save sign bit from fp32 value
            v.si ^= sign;                   // remove sign bit from union
            sign >>= shiftSign;             // shift sign bit from bit 31 to bit 15

            s.si = mulN;
            s.si = s.f * v.f; // correct subnormals
            v.si ^= (s.si ^ v.si) & -(minN > v.si);
            v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
            v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));

            uint32_t roundAddend = v.ui & roundBit ;

            v.ui >>= shift; // logical shift
            v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
            v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);

            // round to nearest if not a special number (Nan, +-infiity)
            if ((v.ui != infC) && (v.ui != NaNC))
            {
                v.ui = v.ui + (roundAddend>>12) ;
            }

            return v.ui | sign;
        }

        static float decompress(uint16_t value)
        {
            Bits v;
            v.ui = value;
            int32_t sign = v.si & signC;
            v.si ^= sign;
            sign <<= shiftSign;
            v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
            v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
            Bits s;
            s.si = mulC;
            s.f *= v.si;
            int32_t mask = -(norC > v.si);
            v.si <<= shift;
            v.si ^= (s.si ^ v.si) & mask;
            v.si |= sign;
            return v.f;
        }
    };


/// @brief Convert fp32 to fp16
/// param[in] x - float(fp32) input to be converted
/// @return fp16 value
//Convert float to fp16 hex value
unsigned int f32Tof16(float flt);
#ifdef __cplusplus
}
#endif


#endif