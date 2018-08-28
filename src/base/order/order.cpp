#include "include/mcm/base/order/order.hpp"

unsigned mv::OrderClass::subToInd(const Shape &s, const std::vector<std::size_t>& sub) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.size() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = firstContiguousDimensionIndex(s); i != -1; i = nextContiguousDimensionIndex(s, i))
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    return currentResult;
}

std::vector<std::size_t> mv::OrderClass::indToSub(const Shape &s, std::size_t idx) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    std::vector<std::size_t> sub(s.ndims());
    sub[firstContiguousDimensionIndex(s)] =  idx % s[firstContiguousDimensionIndex(s)];
    int offset = -sub[firstContiguousDimensionIndex(s)];
    int scale = s[firstContiguousDimensionIndex(s)];
    for (int i = nextContiguousDimensionIndex(s, firstContiguousDimensionIndex(s)); i != -1; i = nextContiguousDimensionIndex(s, i))
    {
        sub[i] = (idx + offset) / scale % s[i];
        offset -= sub[i] * scale;
        scale *= s[i];
    }

    return sub;
}

bool mv::OrderClass::isLastContiguousDimensionIndex(const Shape &s, std::size_t index) const
{
    return index == lastContiguousDimensionIndex(s);
}

bool mv::OrderClass::isFirstContiguousDimensionIndex(const Shape &s, std::size_t index) const
{
    return index == firstContiguousDimensionIndex(s);
}


