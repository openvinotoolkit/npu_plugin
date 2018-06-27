#include "gtest/gtest.h"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(tensor, populating)
{

    mv::Shape tShape(5, 5);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType::Float, mv::Order::LastDimMajor);
    t.populate(data);

    for (unsigned j = 0; j < tShape[0]; ++j)
        for (unsigned i = 0; i < tShape[1]; ++i)
            ASSERT_EQ(t(i, j), i + tShape[0] * j);

}

TEST(tensor, subToInd)
{

    mv::Shape tShape(64, 64, 32, 32, 16);
    std::vector<unsigned> subs[] = {
        {32, 16, 4, 8, 2},
        {15, 62, 31, 31, 0},
        {63, 61, 12, 26, 10},
        {60, 12, 16, 17, 15},
        {0, 15, 0, 5, 4}
    };

    auto idxFcn = [tShape](std::vector<unsigned> s) {
        return s[0] + tShape[0] * (s[1] + tShape[1] * (s[2] + tShape[2] * (s[3] + tShape[3] * s[4])));
    };

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(mv::Tensor::subToInd(tShape, subs[i]),
            idxFcn(subs[i]));


}

TEST(tensor, indToSub)
{

    mv::Shape tShape(64, 64, 32, 32, 16);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType::Float, mv::Order::LastDimMajor);
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};

    for (unsigned i = 0; i < 5; ++i)
    {
        mv::dynamic_vector<unsigned> sub = mv::Tensor::indToSub(tShape, idx[i]);
        ASSERT_EQ(t(sub), idx[i]);
    }

}

TEST(tensor, add)
{

    mv::float_type start = -100.0f;
    mv::float_type diff = 0.5f;

    mv::Shape tShape(32, 32, 3);
    mv::dynamic_vector<mv::float_type> data1 = mv::utils::generateSequence<mv::float_type>(tShape.totalSize(), start, diff);
    mv::dynamic_vector<mv::float_type> data2 = mv::utils::generateSequence<mv::float_type>(tShape.totalSize(), -start, -diff);

    mv::Tensor t1("t1", tShape, mv::DType::Float, mv::Order::LastDimMajor, data1);
    mv::Tensor t2("t2", tShape, mv::DType::Float, mv::Order::LastDimMajor, data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t3(i, j, k), 0.0f);

}

TEST(tensor, add_broadcast_vec)
{

    mv::float_type start = -100.0f;
    mv::float_type diff = 0.5f;

    mv::Shape t1Shape(32, 32, 3);
    mv::Shape t2Shape(3);
    mv::dynamic_vector<mv::float_type> data1 = mv::utils::generateSequence<mv::float_type>(t1Shape.totalSize(), start, diff);
    mv::dynamic_vector<mv::float_type> data2 = mv::utils::generateSequence<mv::float_type>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DType::Float, mv::Order::LastDimMajor, data1);
    mv::Tensor t2("t2", t2Shape, mv::DType::Float, mv::Order::LastDimMajor, data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t1Shape[1]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                ASSERT_FLOAT_EQ(t3(i, j, k), t1(i, j, k) + t2(k));

}

TEST(tensor, add_broadcast_mat)
{

    mv::float_type start = -100.0f;
    mv::float_type diff = 0.5f;

    mv::Shape t1Shape(32, 1, 3);
    mv::Shape t2Shape(16, 3);
    mv::dynamic_vector<mv::float_type> data1 = mv::utils::generateSequence<mv::float_type>(t1Shape.totalSize(), start, diff);
    mv::dynamic_vector<mv::float_type> data2 = mv::utils::generateSequence<mv::float_type>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DType::Float, mv::Order::LastDimMajor, data1);
    mv::Tensor t2("t2", t2Shape, mv::DType::Float, mv::Order::LastDimMajor, data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t2Shape[0]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                ASSERT_FLOAT_EQ(t3(i, j, k), t1(i, 0, k) + t2(j, k));

}

TEST(tensor, add_broadcast_eq)
{

    mv::float_type start = -100.0f;
    mv::float_type diff = 0.5f;

    mv::Shape t1Shape(32, 1, 3, 1);
    mv::Shape t2Shape(32, 3, 1, 16);
    mv::dynamic_vector<mv::float_type> data1 = mv::utils::generateSequence<mv::float_type>(t1Shape.totalSize(), start, diff);
    mv::dynamic_vector<mv::float_type> data2 = mv::utils::generateSequence<mv::float_type>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DType::Float, mv::Order::LastDimMajor, data1);
    mv::Tensor t2("t2", t2Shape, mv::DType::Float, mv::Order::LastDimMajor, data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t2Shape[1]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                for (unsigned l = 0; l < t2Shape[3]; ++l)
                    ASSERT_FLOAT_EQ(t3(i, j, k, l), t1(i, 0, k, 0) + t2(i, j, 0, l));

}

TEST(tensor, subtract)
{

    mv::float_type start = -100.0f;
    mv::float_type diff = 0.5f;

    mv::Shape tShape(32, 32, 3);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DType::Float, mv::Order::LastDimMajor, data);
    mv::Tensor t2("t2", tShape, mv::DType::Float, mv::Order::LastDimMajor, data);

    t1.subtract(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t1(i, j, k), 0.0f);

}

TEST(tensor, multiply)
{

    mv::float_type start = 1.0f;
    mv::float_type diff = 0.5f;

    mv::Shape tShape(32, 32, 3);
    mv::dynamic_vector<mv::float_type> data1 = mv::utils::generateSequence<mv::float_type>(tShape.totalSize(), start, diff);
    mv::dynamic_vector<mv::float_type> data2(data1.size());

    for (unsigned i = 0; i < data2.size(); ++i)
        data2[i] = 1.0f / data1[i];

    mv::Tensor t1("t1", tShape, mv::DType::Float, mv::Order::LastDimMajor, data1);
    mv::Tensor t2("t2", tShape, mv::DType::Float, mv::Order::LastDimMajor, data2);

    t1.mulitply(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t1(i, j, k), 1.0f);

}

TEST(tensor, divide)
{

    mv::float_type start = 2.0f;
    mv::float_type diff = 0.5f;

    mv::Shape tShape(32, 32, 3);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DType::Float, mv::Order::LastDimMajor, data);
    mv::Tensor t2("t2", tShape, mv::DType::Float, mv::Order::LastDimMajor, data);

    t1.divide(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t1(i, j, k), 1.0f);

}