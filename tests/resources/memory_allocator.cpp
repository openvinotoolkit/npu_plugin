#include "mcm/computation/tensor/tensor.hpp"
#include "mcm/computation/resource/memory_allocator.hpp"
#include "mcm/base/order/order_factory.hpp"
#include "mcm/base/order/order.hpp"
#include "gtest/gtest.h"

TEST(memory_allocator, tensor_col_major)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::ColumnMajor;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    mv::dynamic_vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

TEST(memory_allocator, tensor_col_major_planar)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::ColumnMajorPlanar;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    mv::dynamic_vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

TEST(memory_allocator, tensor_row_major)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::RowMajor;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    mv::dynamic_vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

TEST(memory_allocator, tensor_row_major_planar)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::RowMajorPlanar;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    mv::dynamic_vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

