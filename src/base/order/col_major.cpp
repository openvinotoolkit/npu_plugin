#include "mcm/base/order/col_major.hpp"

mv::ColMajor::~ColMajor()
{

}

unsigned mv::ColMajor::subToInd(const Shape &s, const static_vector<dim_type, byte_type, max_ndims>& sub) const
{

    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (unsigned i = 0; i < sub.length(); ++i)
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    return currentResult;

}

mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::ColMajor::indToSub(const Shape &s, unsigned idx) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    static_vector<dim_type, byte_type, max_ndims> sub(s.ndims());
    sub[0] =  idx % s[0];
    int offset = -sub[0];
    int scale = s[0];
    for (int i = 1; i < s.ndims(); ++i)
    {
        sub[i] = (idx + offset) / scale % s[i];
        offset -= sub[i] * s[i - 1];
        scale *= s[i];
    }

    return sub;
}

int mv::ColMajor::previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    return current_dim - 1;
}

int mv::ColMajor::nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    if(current_dim + 1 == s.ndims())
        return -1;
    else
        return current_dim + 1;
}

bool mv::ColMajor::isLastContiguousDimensionIndex(const Shape &s, unsigned index) const
{
    if(index == s.ndims() - 1)
        return true;
    else
        return false;
}

bool mv::ColMajor::isFirstContiguousDimensionIndex(const Shape &s, unsigned index) const
{
    if(index == 0)
        return true;
    else
        return false;
}

//Column major -> First dimension is contiguos -> Last dimension is least contiguous
unsigned mv::ColMajor::lastContiguousDimensionIndex(const Shape &s) const
{
    return s.ndims() - 1;
}

//Column major -> First dimension is contiguos
unsigned mv::ColMajor::firstContiguousDimensionIndex(const Shape &s) const
{
    return 0;
}
