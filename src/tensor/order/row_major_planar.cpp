#include "include/mcm/tensor/order.hpp"

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::rowMajPlanPrevContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)->int
{

    if(s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajorPlanar), "Can't check index for 0-dimensional shape");

    if(dim >= s.ndims())
        throw ShapeError(Order(OrderType::RowMajorPlanar), "Dimension index is bigger than number of dimensions");

    if(dim == 0)
    {
        if(s.ndims() <= 2)
            return -1;
        else
            return 2;
    }

    if(dim == 1)
        return 0;

    if(dim + 1 == s.ndims())
        return -1;
    
    return dim + 1;

};

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::rowMajPlanNextContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)->int
{

    if(s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajorPlanar), "Can't check index for 0-dimensional shape");

    if(dim >= s.ndims())
        throw ShapeError(Order(OrderType::RowMajorPlanar), "Dimension index is bigger than number of dimensions");

    if(dim == 1)
        return -1;

    if(dim == 0)
    {
        if(s.ndims() == 1)
            return -1;
        else
            return 1;
    }

    if(dim == 2)
        return 0;

    return dim - 1;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::rowMajPlanFirstContiguousDimIdx_ =
[](const Shape& s)->std::size_t
{

    if(s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajorPlanar), "0-dimensional shapes have no first contiguous dimension");

    if(s.ndims() <= 2)
        return 0;

    return s.ndims() - 1;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::rowMajPlanLastContiguousDimIdx_ =
[](const Shape& s)
{

    if(s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajorPlanar), "0-dimensional shapes have no first contiguous dimension");

    if(s.ndims() == 1)
        return 0;
    
    return 1;

};