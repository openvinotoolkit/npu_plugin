#include "mcm/base/order/row_major_planar.hpp"

mv::RowMajorPlanar::~RowMajorPlanar()
{

}

int mv::RowMajorPlanar::previousContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
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
    if(current_dim + 1 == s.ndims())
        return -1;
    else
        return current_dim + 1;
}

int mv::RowMajorPlanar::nextContiguousDimensionIndex(const Shape& s, unsigned current_dim) const
{
    if(s.ndims() == 0)
        throw ShapeError("Can't check index for 0-dimensional shape");
    if(current_dim >= s.ndims())
        throw ShapeError("Dimension index is bigger than number of dimensions");
    if(current_dim == 1)
        return -1;
    if(current_dim == 0)
    {
        if(s.ndims() == 1)
            return -1;
        else
            return 1;
    }
    if(current_dim == 2)
        return 0;
    else
    {
        return current_dim - 1;
    }
}

unsigned mv::RowMajorPlanar::lastContiguousDimensionIndex(const Shape &s) const
{
    if(s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    if(s.ndims() == 1)
        return 0;
    else
        return 1;
}

//Planar2 is like RowMajor, only the last; two dimensions are swapped.
unsigned mv::RowMajorPlanar::firstContiguousDimensionIndex(const Shape &s) const
{
    if(s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    if(s.ndims() <= 2)
        return 0;
    else
        return s.ndims() - 1;
}

