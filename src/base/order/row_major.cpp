#include "mcm/base/order/row_major.hpp"

mv::RowMajor::~RowMajor()
{

}

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
