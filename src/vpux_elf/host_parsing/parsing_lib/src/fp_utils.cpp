// {% copyright %}
///
/// @file
/// @copyright All code copyright Movidius Ltd 2014, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Minimal Fp16 conversion software
///            minimal library for fp16 conversions providing helper functions
///            for users sharing data between Leon and SHAVES or other HW blocks
///            which expect fp16 data

#include <fp_utils.h>
#include <mv_types.h>

namespace parsing_lib {

unsigned int rnd_mode;
unsigned int exceptionsReg;
unsigned int* exceptions = &exceptionsReg;

// Constants for converting to BF16
constexpr uint32_t fp32FracBits = 23;
constexpr uint32_t fp16ExpBias = 15;
constexpr uint32_t fp16FracBits = 10;
constexpr uint32_t bf16FracBits = 7;
constexpr uint32_t bf16NanOutput = 0x7FC0;
constexpr uint32_t fp32NanOutput = 0x7FC00000; // Aligns with Synopsys DWC_FP_MULT fixed NAN output

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

//#####################################################################################################
float f16Tof32(unsigned int x)
{
    unsigned int sign;
    int exp;
    unsigned int frac;
    unsigned int result;
    u32f32       u;

    frac = EXTRACT_F16_FRAC(x);
    exp  = EXTRACT_F16_EXP(x);
    sign = EXTRACT_F16_SIGN(x);
    if (exp == 0x1F)
    {
        if (frac != 0)
        {
            // NaN
            if (F16_IS_SNAN(x))
            {
                *exceptions |= F32_EX_INVALID;
            }
            result = 0;
            //Get rid of exponent and sign
#ifndef MOVIDIUS_FP32
            result = x << 22;
            result = f32_shift_right(result, 9);
            result |= ((sign << 31) | 0x7F800000);
#else
            result |= ((sign << 31) | 0x7FC00000);
#endif
        }
        else
        {
            //infinity
            result = PACK_F32(sign, 0xFF, 0);
        }
    }
    else if (exp == 0)
    {
        //either denormal or zero
        if (frac == 0)
        {
            //zero
            result = PACK_F32(sign, 0, 0);
        }
        else
        {
            //subnormal
#ifndef MOVIDIUS_FP32
            f16_normalize_subnormal(&frac, &exp);
            exp--;
            // ALDo: is the value 13 ok??
            result = f16_shift_left(frac, 13);
            // exp = exp + 127 - 15 = exp + 112
            result = PACK_F32(sign, (exp + 0x70), result);
#else
            result = PACK_F32(sign, 0, 0);
#endif
        }
    }
    else
    {
        // ALDo: is the value 13 ok??
        result = f16_shift_left(frac, 13);
        result = PACK_F32(sign, (exp + 0x70), result);
    }

    u.u32 = result;
    return u.f32; //andreil
}

// Apply RNE rounding to fp16 fractional part
// Note if overflow of frac (>10 bits) occurs when rounding up this will
// propagate to the exponent when added in the PACK16 function
uint32_t RoundFp16(uint32_t &dataIn, uint32_t fracWidth) {
    uint32_t frac;
    uint32_t precisionBitsMask;
    uint32_t precisionBits; // Bits used to determine precision
    uint32_t tie;

    if (fracWidth > fp16FracBits) {
        precisionBitsMask = (0x01 << (fracWidth - fp16FracBits)) - 1; // Bits to determine rounding
        precisionBits = dataIn & precisionBitsMask;
        frac = dataIn >> (fracWidth - fp16FracBits); // Pre-rounded fp16 fraction

        tie = 0x01 << (fracWidth - fp16FracBits - 1); // -1 so that we end up with leading 1-bit at MSB of precisionBits
        if (precisionBits > tie) {
            frac++;
        } else if (precisionBits == tie) {
            if ((frac & 0x01)) {
                frac++; // Add 1 if tie and frac is odd (ties to even)
            }
        }
    } else {
        precisionBits = 0;                           // No rounding needed
        frac = dataIn << (fp16FracBits - fracWidth); // fp16 fraction
    }

    return frac;
}

// Taken from ppeQuantisation.cpp in vpu_sysc:
//
// Convert the signed fixed point number to FP16
// Fixed point number is in 2's complement format so need to convert to ((-1)^S)*(1.x1x2...x9)*2^E format
// where: S is the sign
//        x1x2...x9 are the fractional bits after the leading 1-bit
//        E is the biased exponent
int32_t fixedPointToFp16(int32_t x, uint32_t intBits, uint32_t fracBits) {
    uint32_t result;
    uint32_t sign;
    int32_t exp;
    uint32_t frac;

    // Extract the sign and absolute value of x
    sign = (x >> (intBits + fracBits - 1)) & 0x01; // Extract sign bit (assumes signed fixed point input)
    uint32_t xAbs;
    if (sign) {
        xAbs = (~x + 1);
    } else {
        xAbs = x;
    }

    // Detect position of leading 1-bit of input (excluding the sign)
    uint32_t xAbsShift = xAbs;
    uint32_t count = 0;
    while (xAbsShift >>= 1) // Shift right until the leading 1 bit has been shifted off (xAbs becomes false)
    {
        count++;
    }

    // Calculate the fp16 exponent
    // (count - fracBits) is amount of bits shifted relative to fixed point decimal location
    exp = (int32_t(count) - int32_t(fracBits)) + int32_t(fp16ExpBias);

    // Calculate the fp16 fractional part (remaining bits after the leading 1-bit)
    uint32_t xAbsFrac;
    if (count == 0) // Input is zero or denorm
    {
        // Shift frac bits of fixed point input to fill upper bits of fp16 frac
        frac = xAbs << (fp16FracBits - count - 1);
    } else {
        xAbsFrac = xAbs ^ (0x01 << count); // Fractional part excluding leading 1-bit
        frac = RoundFp16(xAbsFrac, count);
    }

    result = (int32_t)PACK_F16(sign, exp, frac);

    return result;
}

// F32 to BFP16 conversion
unsigned int f32_to_b16_conv(unsigned int x, unsigned int rnd_mode, unsigned int *exceptions) {
    unsigned int result;
    unsigned int sign;
    int exp;
    unsigned int frac; //, res_frac;

    frac = EXTRACT_F32_FRAC(x);
    exp = EXTRACT_F32_EXP(x);
    sign = EXTRACT_F32_SIGN(x);

    if (exp == 0xFF) {
        // it's either a NaN or infinite
        if (frac != 0) {
            // NaN
            if (((frac >> 22) & 0x1) == 0x0) {
                // signalling NaN
                *exceptions |= F32_EX_INVALID;
            }
            result = 0x7FC0;
        } else {
            // infinity
            result = PACK_B16(sign, 0xFF, 0);
        }
    } else if (exp == 0x0) {
        if (frac != 0) {
            // Denormal
            // Flush to zero
            *exceptions |= (F32_EX_INEXACT | F32_EX_UNDERFLOW);
            result = PACK_B16(sign, 0, 0);
        } else {
            // Zero
            result = PACK_B16(sign, 0, 0);
        }
    } else {
        // Extract lsb, round and sticky bits
        int lsb = frac & 0x10000;
        int round = frac & 0x8000;
        int sticky = ((frac & 0x7fff) != 0) ? 1 : 0;

        // Truncate significand
        frac = frac >> 16;

        // Increment if necessary
        switch (rnd_mode) {
            case F32_RND_NEAREST_EVEN:
                if ((round && lsb) || (round && sticky))
                    frac = frac + 1;
                break;
            case F32_RND_TO_ZERO:
                break;
            case F32_RND_PLUS_INF:
                if ((sign == 0) && (round || sticky))
                    frac = frac + 1;
                break;
            case F32_RND_MINUS_INF:
                if ((sign == 1) && (round || sticky))
                    frac = frac + 1;
                break;
        }

        // Inexact if either round or sticky bit set
        if (round || sticky)
            *exceptions |= F32_EX_INEXACT;

        // Check if rounding caused significand overflow
        if ((frac & 0x80) != 0) {
            frac = 0;
            exp = exp + 1;
        }

        result = PACK_B16(sign, exp, frac);
    }

    return result;
}
}
