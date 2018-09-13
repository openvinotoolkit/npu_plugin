#include "include/mcm/tensor/order.hpp"

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::rowMajPrevContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)->int
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajor), "0-dimensional shapes have no first contiguous dimension");

    if(dim + 1 == s.ndims())
        return -1;
    
    return dim + 1;

};

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::rowMajNextContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajor), "Cannot check index for 0-dimensional shapes");

    return dim - 1;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::rowMajFirstContiguousDimIdx_ =
[](const Shape& s)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajor), "0-dimensional shapes have no first contiguous dimension");

    return s.ndims() - 1;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::rowMajLastContiguousDimIdx_ =
[](const Shape& s)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowMajor), "0-dimensional shapes have no last contiguous dimension");

    return 0;

};