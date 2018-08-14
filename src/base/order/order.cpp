#include "mcm/base/order/order.hpp"

unsigned mv::OrderClass::subToInd(const Shape &s, const static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
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

mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::OrderClass::indToSub(const Shape &s, unsigned idx) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    static_vector<dim_type, byte_type, max_ndims> sub(s.ndims());
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

bool mv::OrderClass::isLastContiguousDimensionIndex(const Shape &s, unsigned index) const
{
    return index == lastContiguousDimensionIndex(s);
}

bool mv::OrderClass::isFirstContiguousDimensionIndex(const Shape &s, unsigned index) const
{
    return index == firstContiguousDimensionIndex(s);
}


