#include "mcm/base/order/planar.hpp"

mv::Planar::~Planar()
{

}
/*
unsigned mv::Planar::subToInd(const Shape &s, const static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    if (sub.length() != s.ndims())
        throw ShapeError("Mismatch between subscript vector and number of dimensions in shape");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = sub.length() - 1; i > 1; --i)
    {

        if (sub[i] >=  s[i])
            throw ShapeError("Subscript exceeds the dimension");

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    currentResult += currentMul * sub[0];
    currentMul *= s[0];

    if (sub.length() > 1)
        currentResult += currentMul * sub[1];

    return currentResult;
}

mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::Planar::indToSub(const Shape &s, unsigned idx) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot compute subscripts for 0-dimensional shape");

    mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> sub(s.ndims());

    if (s.ndims() == 1)
    {
        sub[0] =  idx % s[0];
        return sub;
    }
    else if (s.ndims() == 2)
    {
        sub[0] = idx % s[0];
        sub[1] = (idx - sub[0]) / s[0] % s[1];
        return sub;
    }
    else
    {
        sub[s.ndims() - 1] =  idx % s[s.ndims() - 1];
        int offset = -sub[s.ndims() - 1];
        int scale = s[s.ndims() - 1];
        for (int i = s.ndims() - 2; i > 1; --i)
        {
            sub[i] = (idx + offset) / scale % s[i];
            offset -= sub[i] * scale;
            scale *= s[i];
        }
        sub[0] = (idx + offset) / scale % s[0];
        offset -= sub[0] * scale;
        scale *= s[0];
        sub[1] = (idx + offset) / scale % s[1];
    }

    return sub;
}
*/

int mv::Planar::previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    if(s.ndims() == 0)
        throw ShapeError("Can't check index for 0-dimensional shape");
    if(current_dim >= s.ndims())
        throw ShapeError("Dimension index is bigger than number of dimensions");
    if(current_dim == 0)
    {
        if(s.ndims() == 1)
            return -1;
        else
            return 1;
    }
    if(current_dim == 1)
        return -1;
    if(current_dim == 2)
        return 0;
    else
        return current_dim - 1;
}

int mv::Planar::nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    if(s.ndims() == 0)
        throw ShapeError("Can't check index for 0-dimensional shape");
    if(current_dim >= s.ndims())
        throw ShapeError("Dimension index is bigger than number of dimensions");
    if(current_dim == 0)
    {
        if(s.ndims() <= 2)
            return -1;
        else
            return 2;
    }
    if(current_dim == 1)
        return 0;
    else
    {
        if(current_dim + 1 == s.ndims())
            return -1;
        else
            return current_dim + 1;
    }
}

unsigned mv::Planar::lastContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no last contiguous dimension");
    if(s.ndims() <= 2)
        return 0;
    else
        return s.ndims() - 1;
}

//Planar is like ColMajor, only the first two dimensions are swapped.
unsigned mv::Planar::firstContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    if(s.ndims() == 1)
        return 0;
    else
        return 1;
}

