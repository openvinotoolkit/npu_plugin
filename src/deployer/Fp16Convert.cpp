
#include "include/mcm/deployer/Fp16Convert.h"

unsigned int rnd_mode;
unsigned int exceptionsReg;
unsigned int* exceptions = &exceptionsReg;


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


#define MOVIDIUS_FP32


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