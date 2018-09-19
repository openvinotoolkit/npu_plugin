#include "mcm/tensor/tensor.hpp"
#include "mcm/computation/resource/memory_allocator.hpp"
#include "mcm/computation/model/data_model.hpp"
#include "mcm/computation/model/op_model.hpp"
#include "mcm/tensor/order.hpp"
#include "mcm/utils/data_generator.hpp"
#include "gtest/gtest.h"

TEST(memory_allocator, tensor_col_major)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape s({4, 4, 2});
    mv::Order order = mv::OrderType::ColumnMajor;
    auto t = dm.defineTensor("testTensor", s, mv::DTypeType::Float16, order, mv::utils::generateSequence<double>(s.totalSize()));
    mv::MemoryAllocator m("m1", 10000, order);
    std::vector<std::size_t> paddings(s.ndims());
    paddings[0] = 1;
    paddings[1] = 1;
    paddings[2] = 1;
    std::vector<std::size_t> strides;
    //m.writeStrides(paddings, s, strides);

    m.allocate(t, 0, paddings);

    for (std::size_t i = 0; i < strides.size(); ++i)
        std::cout << strides[i] << std::endl;

    std::cout << m.bufferBegin(0)->second->toString() << std::endl;

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
    std::vector<std::size_t> strides;
    m.writeStrides(paddings, s, strides);

    for (std::size_t i = 0; i < strides.size(); ++i)
        std::cout << strides[i] << std::endl;

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
    std::vector<std::size_t> strides;
    m.writeStrides(paddings, s, strides);

    for (std::size_t i = 0; i < strides.size(); ++i)
        std::cout << strides[i] << std::endl;

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
    std::vector<std::size_t> strides;
    m.writeStrides(paddings, s, strides);

    for (std::size_t i = 0; i < strides.size(); ++i)
        std::cout << strides[i] << std::endl;

}

