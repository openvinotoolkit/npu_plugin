#include "mcm/base/order/col_major.hpp"

mv::ColMajor::~ColMajor()
{

}

int mv::ColMajor::previousContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot check index for 0-dimensional shapes");
    return current_dim - 1;
}

int mv::ColMajor::nextContiguousDimensionIndex(const Shape& s, std::size_t current_dim) const
{
    if (s.ndims() == 0)
        throw ShapeError("Cannot check index for 0-dimensional shapes");
    if(current_dim + 1 == s.ndims())
        return -1;
    else
        return current_dim + 1;
}

//Column major -> First dimension is contiguos -> Last dimension is least contiguous
std::size_t mv::ColMajor::lastContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    return s.ndims() - 1;
}

//Column major -> First dimension is contiguos
std::size_t mv::ColMajor::firstContiguousDimensionIndex(const Shape &s) const
{
    if (s.ndims() == 0)
        throw ShapeError("0-dimensional shapes have no first contiguous dimension");
    return 0;
}
