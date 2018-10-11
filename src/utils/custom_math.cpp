#include "mcm/utils/custom_math.hpp"
#include <cmath>

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

