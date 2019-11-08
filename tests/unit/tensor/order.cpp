#include "gtest/gtest.h"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/tensor.hpp"

TEST(order, row_major0d)
{
    mv::Shape s(0);
    ASSERT_ANY_THROW(mv::Order order(mv::Order::getRowMajorID(s.ndims())));
}

TEST(order, row_major1d)
{
    mv::Shape s({2});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {0};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, row_major2d)
{
    mv::Shape s({2, 2});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {1, 0};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, row_major3d)
{
    mv::Shape s({2, 2, 2});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {2, 1, 0};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, row_major4d)
{
    mv::Shape s({2, 2, 2, 2});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {3, 2, 1, 0};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, row_major5d)
{
    mv::Shape s({2, 2, 2, 2, 2});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {4, 3, 2, 1, 0};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, col_major0d)
{
    mv::Shape s(0);
    ASSERT_ANY_THROW(mv::Order order(mv::Order::getColMajorID(s.ndims())));
}

TEST(order, col_major1d)
{
    mv::Shape s({2});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {0};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, col_major2d)
{
    mv::Shape s({2, 2});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {0, 1};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, col_major3d)
{
    mv::Shape s({2, 2, 2});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {0, 1, 2};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, col_major4d)
{
    mv::Shape s({2, 2, 2, 2});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {0, 1, 2, 3};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, col_major5d)
{
    mv::Shape s({2, 2, 2, 2, 2});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    std::vector<std::size_t> expectedOrder = {0, 1, 2, 3, 4};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, row_interleaved)
{
    mv::Shape s({2, 2, 2});
    mv::Order order("HCW");

    std::vector<std::size_t> expectedOrder = {0, 2, 1};

    for(unsigned i = 0; i < order.size(); ++i)
        ASSERT_EQ(order[i], expectedOrder[i]);
}

TEST(order, tensor_mismatching_order)
{
    mv::Shape s({2, 2});
    mv::Order order("HCW");

    ASSERT_ANY_THROW(mv::Tensor("Test", s, mv::DType("Float16"), order));
}

TEST(order, strides_computation)
{
    mv::Shape s({224, 224, 3});
    mv::Order order1("WHC");
    mv::Order order2("CHW");

    auto result = order1.computeWordStrides(s);
    for(auto x : result)
        std::cout << x << std::endl;

    result = order2.computeWordStrides(s);
    for(auto x : result)
        std::cout << x << std::endl;
}
