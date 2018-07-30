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

unsigned int rnd_mode;
unsigned int exceptionsReg;
unsigned int* exceptions = &exceptionsReg;

//#####################################################################################################
unsigned int f16_shift_left(unsigned int op, unsigned int cnt)
{
    unsigned int result;
    if (cnt == 0)
    {
        result = op;
    }
    else if (cnt < 32)
    {
        result = (op << cnt);
    }
    else
    {
        result = 0;
    }
    return result;
}

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
//#####################################################################################################
//Convert float to fp16 hex value
unsigned int f32Tof16(float flt)
{
#ifdef MOVIDIUS_FP32
    unsigned int s;
    signed int   e;
    unsigned int f;

    u32f32  u;
    u.f32 = flt;
    unsigned int x = u.u32;//andreil

    // clear flags
    *exceptions = 0;
    unsigned int round;
    unsigned int sticky;
    unsigned int flsb;
    // Extract fields
    s = (x >> 31) & 0x00000001;
    e = (x >> 23) & 0x000000ff;
    f = x & 0x007fffff;
    if((e == 0) && (f == 0))
    {
        // fp32 number is zero 
        return(s<<15);
    }
    else if( e==255 )
    {
        if( f==0 )
        {
            // fp32 number is an infinity

            return((s<<15)|0x7c00);
        }
        else

        {
            // fp32 number is a NaN - return QNaN, raise invalid if
            // SNaN. QNaN assumed to have MSB of significand set
            if( ~(f&0x00400000) ) *exceptions |= F32_EX_INVALID;
            return(0x7e00); // Sign of NaN ignored
        }
    }
    else
    {
        // fp32 number is normal or possibly denormal
        // Add hidden bit if normal
        if(e!=0)
        {
            f = f | 0x00800000;
        }
        // Unbias exponent
        e = e-127;
        // Check if not below fp16 normal
        if( e>=-14 ) {
            // Round significand according to specified mode
            // Extract round and sticky bits
            round = (f & 0x00001000) >> 12;
            //sticky = |(f & 0x00000fff);
            //replaced with:
            sticky = ((f & 0x00000fff) == 0) ? 0 : 1;
            // Truncate signficand
            f = f >> 13;
            flsb = f & 0x00000001; // LSB 
            // Increment if necessary 
 
            switch(rnd_mode)
            {
                // Use softfloat mappings (P_CFG will have been mapped before call to CMU
                case F32_RND_NEAREST_EVEN:
                    if((round && flsb) || (round && sticky))
                    {
                        f = f+1;
                    }
                    break;
                case F32_RND_TO_ZERO:
                    break;
                case F32_RND_PLUS_INF:
                    if((s == 0) && (round || sticky))
                    {
                        f = f+1;
                    }
                    break;
                case F32_RND_MINUS_INF:
                    if((s == 1) && (round || sticky))
                    {
                        f = f+1;
                    }
                    break;
            }
            // Inexact if either round or sticky bit set
            if(round || sticky)
            {
                *exceptions |= F32_EX_INEXACT;
            }
            // Check if significand overflow occurred

            if(f&0x00000800)
            {
                f = f >> 1;
                e = e + 1;
            }
            // Add fp16 bias to exponent
            e = e + 15;
            // Check for exponent overflow

            if(e > 30)
            {
                // Return according to rounding mode and set overflow and inexact flags
                *exceptions |=  F32_EX_OVERFLOW;
                *exceptions |=  F32_EX_INEXACT ;
                switch(rnd_mode)
                {
                case F32_RND_NEAREST_EVEN:
                    return ((s << 15) | 0x7c00);// Returns infinity
                case F32_RND_TO_ZERO:
                    return ((s << 15) | 0x7bff);// Largest finite #
                case F32_RND_PLUS_INF:
                    return ((s == 0) ? 0x7c00 : 0xfbff);
                case F32_RND_MINUS_INF:
                    return ((s == 1) ? 0xfc00 : 0x7bff);
                }
            }
            else
            {
                // Remove hidden bit, pack, and return
                f = f & 0x000003ff;
                return ((s << 15) | (e << 10) | f);
            }
        }
        else
        {
            // fp32 number may be representable as a fp16 denormal
            // flushing FP16 denormal outputs to zero
            return(s << 15);
        }
    }
    return 0;
#else
    unsigned int result;
    unsigned int sign;
    int exp;
    unsigned int frac, res_frac;

    frac = EXTRACT_F32_FRAC(x);
    exp  = EXTRACT_F32_EXP(x);
    sign = EXTRACT_F32_SIGN(x);

    if (exp == 0xFF)
    {
        // it's either a NaN or infinite
        if (frac != 0)
        {
            //NaN
            if (((frac >> 22) & 0x1) == 0x0)
            {
                // signalling NaN

                *exceptions |= F32_EX_INVALID;
            }
            result = (sign << 15) | 0x7C00 | (frac >> 13);
        }
        else
        {
            //infinity
            result = PACK_F16(sign, 0x1F, 0);
        }
    }
    else
    {
        // we need to shift 13 positions but will shift only 9, to keep the point at bits 13-14
        res_frac = f32_shift_right_loss_detect(frac, 9);
        //If exponent is not 0, add the implicit 1
        if (exp)
        {
            res_frac |= 0x00004000;
        }
        // exp = exp - 127 + 15 - 1 = exp - 113       
        // -1 -> exponent must be 1 unit less than real exponent for rounding and packing
        result = f16_pack_round(sign, exp - 0x71, res_frac, rnd_mode, exceptions);
    }
    return result;
#endif
}

/// @}
#ifdef __cplusplus
}
#endif


#endif