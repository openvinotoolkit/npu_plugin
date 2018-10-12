#include "include/mcm/tensor/order.hpp"
#include <assert.h>

/*
* All Methods only working for 3D Shapes Only
*/

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::RowInterleaved_PrevContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)->int
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowInterleaved), "0-dimensional shapes have no first contiguous dimension");

    if (dim == 0)
        return -1;
    else if (dim == 1)
        return 2;
    else if (dim == 2)
        return 0;

    throw OrderError(Order(OrderType::RowInterleaved), "RowInterleaved only supports 3D");
    return 0;

};

const std::function<int(const mv::Shape&, std::size_t)> mv::Order::RowInterleaved_NextContiguousDimIdx_ =
[](const Shape& s, std::size_t dim)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowInterleaved), "Cannot check index for 0-dimensional shapes");

    if (dim == 0)
        return 2;
    else if (dim == 1)
        return -1;
    else if (dim == 2)
        return 1;

    throw OrderError(Order(OrderType::RowInterleaved), "RowInterleaved only supports 3D");
    return 0;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::RowInterleaved_FirstContiguousDimIdx_ =
[](const Shape& s)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowInterleaved), "0-dimensional shapes have no first contiguous dimension");

    return 0;

};

const std::function<std::size_t(const mv::Shape&)> mv::Order::RowInterleaved_LastContiguousDimIdx_ =
[](const Shape& s)
{

    if (s.ndims() == 0)
        throw OrderError(Order(OrderType::RowInterleaved), "0-dimensional shapes have no last contiguous dimension");

    return 1;

};