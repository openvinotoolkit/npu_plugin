#include "mcm/utils/custom_math.hpp"
#include <cmath>
#include <algorithm>

unsigned mv::round_up(unsigned x, unsigned mult)
{
    return ((x + mult - 1) / mult) * mult; //power of integer arithmetic, don't touch
}

unsigned mv::ceil_division(unsigned x, unsigned d)
{
    return (x + d - 1) / d;
}

unsigned mv::count_bits(unsigned number)
{
    unsigned bits;
    for(bits = 0; number != 0; ++bits)
        number >>= 1;
    return bits;
}

unsigned mv::next_greater_power_of_2(unsigned number)
{
    return pow(2,count_bits(--number));
}

uint16_t mv::fp32_to_fp16(float value)
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
            v.ui = v.ui + (roundAddend>>12) ;

        return v.ui | sign;
}

uint16_t mv::fp32_to_fp16(double value)
{
    return fp32_to_fp16(static_cast<float>(value));
}

float mv::fp16_to_fp32(uint16_t value){
    bit_field32 v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    bit_field32 s;
    s.si = mulC;
    s.fp *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shiftFraction;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.fp;
}

std::vector<std::size_t> mv::tileSpatialOutputSize(std::size_t outputSize , std::size_t numberOfSplits)
{
    // aim is to get the splits such that the last split is smallest and rest of the splits are equal
    // but if thats not possible, logic is added to have the first one largest and rest of the splits equal (is this right?)
    int newOutputSize = ceil( (double)(outputSize) / (double)numberOfSplits);
    int remainderOutputSize = outputSize - (newOutputSize *(numberOfSplits -1));
    if (remainderOutputSize <= 0)
    {
        newOutputSize = trunc( (double)(outputSize) / (double)numberOfSplits);
        remainderOutputSize = outputSize - (newOutputSize *(numberOfSplits -1));
    }
    std::vector<std::size_t> outputSizes(numberOfSplits, newOutputSize);
    if (remainderOutputSize > newOutputSize)
       outputSizes[0] = remainderOutputSize;
    else
       outputSizes[numberOfSplits-1] = remainderOutputSize; 
    return outputSizes;
}

uint16_t mv::getWindowSize(uint16_t kx, uint16_t sx, mv::DType dataType)
{
    //Find max mpe where if calc window <= 32
    //return window size for the max mpe
    uint16_t windowSize, maxMpeWindowSize = 64;
    int mpe = 1;

    if (dataType == mv::DType("UInt8"))
    {
        //mpe in [1,2,4,8,16] for uint8
        while(mpe <= 16)
        {
            if (sx <= kx)
                windowSize = kx + sx * (mpe - 1);
            else
                windowSize = kx * mpe;

            if (windowSize <= 32)
                maxMpeWindowSize = windowSize;

            mpe *= 2;
        }
    }
    else if (dataType == mv::DType("Float16"))
    {
        //mpe in [1,2,4] for float
        while(mpe <= 4)
        {
            if (sx <= kx)
                windowSize = kx + sx * (mpe - 1);
            else
                windowSize = kx * mpe;

            if (windowSize <= 32)
                maxMpeWindowSize = windowSize;

            mpe *= 2;
        }
    }

    return maxMpeWindowSize;
}
