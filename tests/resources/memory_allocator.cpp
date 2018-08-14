#include "mcm/computation/tensor/tensor.hpp"
#include "mcm/computation/resource/memory_allocator.hpp"
#include "mcm/base/order/order_factory.hpp"
#include "mcm/base/order/order.hpp"
#include "gtest/gtest.h"

TEST(memory_allocator, tensor_colMajor)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::ColumnMajor;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::dynamic_vector<unsigned> paddings;
    paddings.push_back(5);
    paddings.push_back(6);
    paddings.push_back(3);
    mv::dynamic_vector<unsigned> strides;
    std::unique_ptr<mv::OrderClass> orderClass = mv::OrderFactory::createOrder(order);
    m.recursiveWriteStrides(orderClass->lastContiguousDimensionIndex(t.getShape()), paddings, strides, t.getShape());
    std::cout << "Test ended" << std::endl;
}

TEST(memory_allocator, tensor_rowMajor)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::RowMajor;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::dynamic_vector<unsigned> paddings;
    paddings.push_back(5);
    paddings.push_back(6);
    paddings.push_back(3);
    mv::dynamic_vector<unsigned> strides;
    std::unique_ptr<mv::OrderClass> orderClass = mv::OrderFactory::createOrder(order);
    m.recursiveWriteStrides(orderClass->lastContiguousDimensionIndex(t.getShape()), paddings, strides, t.getShape());
    std::cout << "Test ended" << std::endl;
}

TEST(memory_allocator, tensor_planar)
{
    mv::Shape s(3, 2, 5);
    mv::Order order = mv::Order::ColumnMajorPlanar;
    mv::Tensor t("test_tensor", s, mv::DType::Float, order);
    mv::MemoryAllocator m("m1", 10000, order);
    mv::dynamic_vector<unsigned> paddings;
    paddings.push_back(5);
    paddings.push_back(6);
    paddings.push_back(3);
    mv::dynamic_vector<unsigned> strides;
    std::unique_ptr<mv::OrderClass> orderClass = mv::OrderFactory::createOrder(order);
    m.recursiveWriteStrides(orderClass->lastContiguousDimensionIndex(t.getShape()), paddings, strides, t.getShape());
    std::cout << "Test ended" << std::endl;
}
