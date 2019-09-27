#include "mcm/tensor/tensor.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/computation/resource/memory_allocator.hpp"
#include "mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/utils/data_generator.hpp"
#include "gtest/gtest.h"


TEST(memory_allocator, concatenate_tensors)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape outputShape({4, 4, 4});
    mv::Shape inputShape({4, 4, 2});
    mv::Order order(mv::Order::getColMajorID(3));
    
    auto outputTensor = dm.defineTensor("outputTensor", outputShape, mv::DType("Float16"), order,
        mv::utils::generateSequence<double>(outputShape.totalSize()));

    auto inputTensor1 = dm.defineTensor("inputTensor1", inputShape, mv::DType("Float16"), order);
    auto inputTensor2 = dm.defineTensor("inputTensor2", inputShape, mv::DType("Float16"), order);

    mv::MemoryAllocator m("m1", 10000, 0, 2);
    auto outputBuf = m.allocate(outputTensor, 0);
    auto input1Buf = m.allocate(inputTensor1, outputBuf, {0, 0, 0}, {0, 0, 2});
    auto input2Buf = m.allocate(inputTensor2, outputBuf, {0, 0, 2}, {0, 0, 0});

    std::cout << (*outputBuf)->toString(true) << std::endl;
    std::cout << (*input1Buf)->toString(true) << std::endl;
    std::cout << (*input2Buf)->toString(true) << std::endl;

    std::cout << m.toString() << std::endl;

}

TEST(memory_allocator, mulitple)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape s1({4, 4, 4});
    mv::Shape s2({4, 8, 2});
    mv::Shape s3({32, 2, 2});
    mv::Order order = mv::Order("CHW");
    
    auto t1 = dm.defineTensor("t1", s1, mv::DType("Float16"), order);
    auto t2 = dm.defineTensor("a2", s2, mv::DType("Float16"), order);
    auto t3 = dm.defineTensor("t3", s3, mv::DType("Float16"), order);

    mv::MemoryAllocator m("m1", 10000, 0, 2);
    //auto b1 = m.allocate(t1, 0);
    //auto b2 = m.allocate(t2, 0);
    //auto b3 = m.allocate(t3, 0);

    //std::cout << m.toString() << std::endl;

}

TEST(memory_allocator, tensor_col_major)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape s({2, 2});
    mv::Order order = mv::Order(mv::Order::getRowMajorID(2));;
    auto t = dm.defineTensor("testTensor", s, mv::DType("Float16"), order, mv::utils::generateSequence<double>(s.totalSize()));

    mv::MemoryAllocator m("m1", 10000, 0, 2);
    std::vector<std::size_t> padding1(s.ndims()), padding2(s.ndims());
    padding1[0] = 1;
    padding1[1] = 1;
    padding2[0] = 2;
    padding2[1] = 2;

    auto buf = m.allocate(t, 0);
    m.padLeft(buf, padding1);

    for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << (*it)->toString(true) << std::endl;

    /**m.padLeft(buf, padding2);

    for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << it->second->toString(true) << std::endl;*/

    m.padRight(buf, padding2);

    /*for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << it->second->toString(true) << std::endl;

    m.padRight(buf, padding2);*/

    for (auto it = m.bufferBegin(0); it != m.bufferEnd(0); ++it)
        std::cout << (*it)->toString(true) << std::endl;
    
}

TEST(memory_allocator, slave_tensor_col_major)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape masterShape({4, 4});
    mv::Shape slaveShape({2, 2});
    mv::Order order = mv::Order(mv::Order::getRowMajorID(2));;
    auto tMaster = dm.defineTensor("masterTensor", masterShape, mv::DType("Float16"), order,
        mv::utils::generateSequence<double>(masterShape.totalSize()));
    auto tSlave = dm.defineTensor("slaveShape", slaveShape, mv::DType("Float16"), order,
        mv::utils::generateSequence<double>(slaveShape.totalSize()));


    mv::MemoryAllocator m("m1", 10000, 0, 2);
    auto masterBuf = m.allocate(tMaster, 0);
    auto slaveBuf = m.allocate(tSlave, masterBuf, {0, 0}, {2, 2});
    
    std::cout << (*masterBuf)->toString(true) << std::endl;
    std::cout << (*slaveBuf)->toString(true) << std::endl;

    std::cout << tMaster->toString() << std::endl;
    std::cout << tSlave->toString() << std::endl;

    for (unsigned i = 0; i < tSlave->getShape().totalSize(); ++i)
        std::cout << (std::string) tSlave->at(i) << std::endl;

    tSlave->at({1, 1}) = 30.0;

    std::cout << (*masterBuf)->toString(true) << std::endl;
    std::cout << (*masterBuf)->toString(true) << std::endl;

}

TEST(memory_allocator, move_in_place)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape outputShape({4, 4, 2});
    mv::Shape inputShape({4, 4, 2});
    mv::Order order = mv::Order("CHW");

    auto inputTensor = dm.defineTensor("inputTensor", inputShape, mv::DType("Float16"), order);
    auto outputTensor = dm.defineTensor("outputTensor", outputShape, mv::DType("Float16"), order,
        mv::utils::generateSequence<double>(outputShape.totalSize()));

    mv::MemoryAllocator m("m1", 10000, 0, 2);
    auto inputBuf = m.allocate(inputTensor, 0);
    auto outputBuf = m.allocate(outputTensor, 0);
    m.move(inputBuf, outputBuf, {0, 0, 0}, {0, 0, 0});

    std::cout << (*outputBuf)->toString(true) << std::endl;
    std::cout << (*inputBuf)->toString(true) << std::endl;

    std::cout << m.toString() << std::endl;

}

TEST(memory_allocator, move_concat)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape outputShape({4, 4, 4});
    mv::Shape inputShape({4, 4, 2});
    mv::Order order = mv::Order("CHW");

    auto input1Tensor = dm.defineTensor("inputTensor1", inputShape, mv::DType("Float16"), order);
    auto input2Tensor = dm.defineTensor("inputTensor2", inputShape, mv::DType("Float16"), order);
    auto outputTensor = dm.defineTensor("outputTensor", outputShape, mv::DType("Float16"), order,
        mv::utils::generateSequence<double>(outputShape.totalSize()));

    mv::MemoryAllocator m("m1", 10000, 0, 2);
    auto input1Buf = m.allocate(input1Tensor, 0);
    auto input2Buf = m.allocate(input2Tensor, 0);
    auto outputBuf = m.allocate(outputTensor, 0);
    m.move(input1Buf, outputBuf, {0, 0, 2}, {0, 0, 0});
    m.move(input2Buf, outputBuf, {0, 0, 0}, {0, 0, 2});
    
    std::cout << (*outputBuf)->toString(true) << std::endl;
    std::cout << (*input1Buf)->toString(true) << std::endl;
    std::cout << (*input2Buf)->toString(true) << std::endl;

    std::cout << m.toString() << std::endl;

}

TEST(memory_allocator, move_concat_in_place)
{

    mv::OpModel om("testModel");
    mv::DataModel dm(om);
    mv::Shape outputShape({4, 4, 4});
    mv::Shape inputShape({4, 4, 2});
    mv::Order order = mv::Order("CHW");

    auto input1Tensor = dm.defineTensor("inputTensor1", inputShape, mv::DType("Float16"), order);
    auto input2Tensor = dm.defineTensor("inputTensor2", inputShape, mv::DType("Float16"), order);
    auto inter1Tensor = dm.defineTensor("interTensor1", inputShape, mv::DType("Float16"), order);
    auto inter2Tensor = dm.defineTensor("interTensor2", inputShape, mv::DType("Float16"), order);
    auto outputTensor = dm.defineTensor("outputTensor", outputShape, mv::DType("Float16"), order,
        mv::utils::generateSequence<double>(outputShape.totalSize()));

    mv::MemoryAllocator m("m1", 10000, 0, 2);
    auto input1Buf = m.allocate(input1Tensor, 0);
    auto input2Buf = m.allocate(input2Tensor, 0);
    auto inter1Buf = m.allocate(inter1Tensor, 0);
    auto inter2Buf = m.allocate(inter2Tensor, 0);
    auto outputBuf = m.allocate(outputTensor, 0);

    std::cout << input1Tensor->getShape().toString() << std::endl;
    std::cout << input2Tensor->getShape().toString() << std::endl;
    std::cout << inter1Tensor->getShape().toString() << std::endl;
    std::cout << inter2Tensor->getShape().toString() << std::endl;
    std::cout << outputTensor->getShape().toString() << std::endl;

    m.move(input1Buf, inter1Buf, {0, 0, 0}, {0, 0, 0});
    m.move(input2Buf, inter2Buf, {0, 0, 0}, {0, 0, 0});

    std::cout << input1Tensor->getShape().toString() << std::endl;
    std::cout << input2Tensor->getShape().toString() << std::endl;
    std::cout << inter1Tensor->getShape().toString() << std::endl;
    std::cout << inter2Tensor->getShape().toString() << std::endl;
    std::cout << outputTensor->getShape().toString() << std::endl;

    m.move(inter1Buf, outputBuf, {0, 0, 2}, {0, 0, 0});
    m.move(inter2Buf, outputBuf, {0, 0, 0}, {0, 0, 2});
    
    std::cout << input1Tensor->getShape().toString() << std::endl;
    std::cout << input2Tensor->getShape().toString() << std::endl;
    std::cout << inter1Tensor->getShape().toString() << std::endl;
    std::cout << inter2Tensor->getShape().toString() << std::endl;
    std::cout << outputTensor->getShape().toString() << std::endl;

    std::cout << (*outputBuf)->toString(true) << std::endl;
    std::cout << (*input1Buf)->toString(true) << std::endl;
    std::cout << (*input2Buf)->toString(true) << std::endl;
    std::cout << (*inter1Buf)->toString(true) << std::endl;
    std::cout << (*inter2Buf)->toString(true) << std::endl;

    std::cout << m.toString() << std::endl;

}

/*
TEST(memory_allocator, tensor_col_major_planar)
{

    mv::Shape s({3, 2, 5});
    mv::Order order = mv::Order("CHW");
    mv::Tensor t("test_tensor", s, mv::DType("Float16"), order);
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
    mv::Order order = mv::Order(Order::getRowMajorID(3));;
    mv::Tensor t("test_tensor", s, mv::DType("Float16"), order);
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
    mv::Order order = mv::Order("HWC");
    mv::Tensor t("test_tensor", s, mv::DType("Float16"), order);
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
