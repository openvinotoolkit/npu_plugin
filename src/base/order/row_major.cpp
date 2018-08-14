#include "mcm/base/order/row_major.hpp"

mv::RowMajor::~RowMajor()
{

}
/*
unsigned mv::RowMajor::subToInd(const Shape &s, const mv::static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = sub.length() - 1; i >= 0 ; --i)
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    return currentResult;
}

mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::RowMajor::indToSub(const Shape &s, unsigned idx) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> sub(s.ndims());
    sub[s.ndims() - 1] =  idx % s[s.ndims() - 1];
    int offset = -sub[s.ndims() - 1];
    int scale = s[s.ndims() - 1];
    for (int i = s.ndims() - 2; i >= 0; --i)
    {
        sub[i] = (idx + offset) / scale % s[i];
        offset -= sub[i] * scale;
        scale *= s[i];
    }

    return sub;
}

*/

int mv::RowMajor::previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    if(current_dim + 1 == s.ndims())
        return -1;
    else
        return current_dim + 1;
}

int mv::RowMajor::nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot check index for 0-dimensional shapes");
    return current_dim - 1;
}

//RowMajor -> Last dimension is contiguos -> First dimension is least contiguous
unsigned mv::RowMajor::lastContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no last contiguous dimension");
    return 0;
}

//RowMajor -> Last dimension is contiguos
unsigned mv::RowMajor::firstContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    return s.ndims() - 1;
}
