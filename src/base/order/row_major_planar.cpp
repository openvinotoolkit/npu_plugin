#include "mcm/base/order/col_major_planar.hpp"

mv::ColumnMajorPlanar::~ColumnMajorPlanar()
{

}

int mv::ColumnMajorPlanar::previousContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const
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

int mv::ColumnMajorPlanar::nextContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const
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

std::size_t mv::ColumnMajorPlanar::lastContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no last contiguous dimension");
    if(s.ndims() <= 2)
        return 0;
    else
        return s.ndims() - 1;
}

//Planar is like ColMajor, only the first two dimensions are swapped.
std::size_t mv::ColumnMajorPlanar::firstContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    if(s.ndims() == 1)
        return 0;
    else
        return 1;
}

