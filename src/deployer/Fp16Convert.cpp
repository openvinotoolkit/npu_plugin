
#include "include/mcm/utils/serializer/Fp16Convert.h"

uint16_t mv_num_convert::fp32_to_fp16(float value)
{
        bit_field32 v, s;                   // operate on a 32 bit union of types
        v.fp = value;                       // original FP32 value to convert

        unsigned int sign = v.si & sign32;  // save sign bit from fp32 value
        v.si ^= sign;                       // remove sign bit from union
        sign >>= shiftSign;                 // hold sign bit shifted from bit 31 to bit 15

        s.si = mulN;
        s.si = s.fp * v.fp; // correct subnormals
        v.si ^= (s.si ^ v.si) & -(minNorm > v.si);
        v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxNorm));
        v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));

        unsigned int roundAddend = v.ui & roundBit ;

        v.ui >>= shiftFraction; // logical shift
        v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
        v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);

        // round to nearest if not a special number (Nan, +-infiity)
        if (v.ui < inf16)
        {
            v.ui = v.ui + (roundAddend>>12) ;
        }

        return v.ui | sign;
}

uint16_t mv_num_convert::fp32_to_fp16(double value)
{

    return fp32_to_fp16(static_cast<float>(value));

}