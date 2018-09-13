#include "mcm/computation/tensor/tensor.hpp"
#include "mcm/computation/resource/memory_allocator.hpp"
#include "mcm/computation/tensor/order.hpp"
#include "gtest/gtest.h"

TEST(memory_allocator, tensor_col_major)
{
    mv::Shape s({3, 2, 5});
    mv::Order order = mv::OrderType::ColumnMajor;
    mv::Tensor t("test_tensor", s, mv::DTypeType::Float16, order);
    mv::MemoryAllocator m("m1", 10000, order);
    std::vector<std::size_t> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    std::vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

TEST(memory_allocator, tensor_col_major_planar)
{
    mv::Shape s({3, 2, 5});
    mv::Order order = mv::OrderType::ColumnMajorPlanar;
    mv::Tensor t("test_tensor", s, mv::DTypeType::Float16, order);
    mv::MemoryAllocator m("m1", 10000, order);
   std::vector<std::size_t> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    std::vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

TEST(memory_allocator, tensor_row_major)
{
    mv::Shape s({3, 2, 5});
    mv::Order order = mv::OrderType::RowMajor;
    mv::Tensor t("test_tensor", s, mv::DTypeType::Float16, order);
    mv::MemoryAllocator m("m1", 10000, order);
    std::vector<std::size_t> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    std::vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

TEST(memory_allocator, tensor_row_major_planar)
{
    mv::Shape s({3, 2, 5});
    mv::Order order = mv::OrderType::RowMajorPlanar;
    mv::Tensor t("test_tensor", s, mv::DTypeType::Float16, order);
    mv::MemoryAllocator m("m1", 10000, order);
    std::vector<std::size_t> paddings(s.ndims());
    paddings[0] = 5;
    paddings[1] = 6;
    paddings[2] = 3;
    std::vector<unsigned> strides;
    m.writeStrides(paddings, s, strides);
}

