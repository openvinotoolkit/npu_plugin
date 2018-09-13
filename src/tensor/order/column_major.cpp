#include "include/mcm/tensor/order.hpp"

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::colMajPrevContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::ColumnMajor), "Cannot check index for 0-dimensional shapes");

    return dim - 1;

};

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::colMajNextContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)->int
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::ColumnMajor), "Cannot check index for 0-dimensional shapes");

    if(dim + 1 == s.ndims())
        return -1;

    return dim + 1;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::colMajFirstContiguousDimIdx_ =
[](const Shape& s)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::ColumnMajor), "0-dimensional shapes have no first contiguous dimension");

    return 0;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::colMajLastContiguousDimIdx_ =
[](const Shape& s)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::ColumnMajor), "0-dimensional shapes have no first contiguous dimension");

    return s.ndims() - 1;

};