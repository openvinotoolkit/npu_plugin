#include "mcm/tensor/tensor.hpp"
#include "mcm/computation/resource/memory_allocator.hpp"
#include "mcm/computation/model/data_model.hpp"
#include "mcm/computation/model/op_model.hpp"
#include "mcm/tensor/order.hpp"
#include "mcm/utils/data_generator.hpp"
#include "gtest/gtest.h"


TEST(memory_allocator, concatenate_tensors)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape outputShape({4, 4, 4});
    mv::Shape inputShape({4, 4, 2});
    mv::Order order = mv::OrderType::ColumnMajor;
    
    auto outputTensor = dm.defineTensor("outputTensor", outputShape, mv::DTypeType::Float16, order,
        mv::utils::generateSequence<double>(outputShape.totalSize()));

    auto inputTensor1 = dm.defineTensor("inputTensor1", inputShape, mv::DTypeType::Float16, order);
    auto inputTensor2 = dm.defineTensor("inputTensor2", inputShape, mv::DTypeType::Float16, order);

    mv::MemoryAllocator m("m1", 10000);
    auto outputBuf = m.allocate(outputTensor, 0);
    auto input1Buf = m.allocate(inputTensor1, outputBuf, {0, 0, 0}, {0, 0, 2});
    auto input2Buf = m.allocate(inputTensor2, outputBuf, {0, 0, 2}, {0, 0, 0});

    std::cout << outputBuf->second->toString(true) << std::endl;
    std::cout << input1Buf->second->toString(true) << std::endl;
    std::cout << input2Buf->second->toString(true) << std::endl;

}

TEST(memory_allocator, tensor_col_major)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape s({2, 2});
    mv::Order order = mv::OrderType::RowMajor;
    auto t = dm.defineTensor("testTensor", s, mv::DTypeType::Float16, order, mv::utils::generateSequence<double>(s.totalSize()));

    mv::MemoryAllocator m("m1", 10000);
    std::vector<std::size_t> padding1(s.ndims()), padding2(s.ndims());
    padding1[0] = 1;
    padding1[1] = 1;
    padding2[0] = 2;
    padding2[1] = 2;

    auto buf = m.allocate(t, 0);
    m.padLeft(buf, padding1);

    for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << it->second->toString(true) << std::endl;

    /**m.padLeft(buf, padding2);

    for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << it->second->toString(true) << std::endl;*/

    m.padRight(buf, padding2);

    /*for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << it->second->toString(true) << std::endl;

    m.padRight(buf, padding2);*/

    for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << it->second->toString(true) << std::endl;
    
}

TEST(memory_allocator, slave_tensor_col_major)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape masterShape({4, 4});
    mv::Shape slaveShape({2, 2});
    mv::Order order = mv::OrderType::RowMajor;
    auto tMaster = dm.defineTensor("masterTensor", masterShape, mv::DTypeType::Float16, order, 
        mv::utils::generateSequence<double>(masterShape.totalSize()));
    auto tSlave = dm.defineTensor("slaveShape", slaveShape, mv::DTypeType::Float16, order,
        mv::utils::generateSequence<double>(slaveShape.totalSize()));


    mv::MemoryAllocator m("m1", 10000);
    auto masterBuf = m.allocate(tMaster, 0);
    auto slaveBuf = m.allocate(tSlave, masterBuf, {0, 0}, {2, 2});
    
    std::cout << masterBuf->second->toString(true) << std::endl;
    std::cout << slaveBuf->second->toString(true) << std::endl;

    std::cout << tMaster->toString() << std::endl;
    std::cout << tSlave->toString() << std::endl;

    for (unsigned i = 0; i < tSlave->getShape().totalSize(); ++i)
        std::cout << tSlave->at(i) << std::endl;

    tSlave->at({1, 1}) = 30.0;
    
    std::cout << masterBuf->second->toString(true) << std::endl;
    std::cout << slaveBuf->second->toString(true) << std::endl;

}

/*
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

*/