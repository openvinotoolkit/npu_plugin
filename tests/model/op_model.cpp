#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

TEST(op_model, validity)
{

    mv::OpModel om("TestModel");
    ASSERT_FALSE(om.isValid());
    auto input = om.input({32, 32, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    ASSERT_FALSE(om.isValid());
    om.output(input);
    ASSERT_TRUE(om.isValid());

}

TEST(op_model, failure_sanity)
{

    mv::OpModel om("TestModel");

    auto input = om.input({32, 32, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    om.output(input);

    //ASSERT_TRUE(om.addAttr(om.getSourceOp(input), "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10)));
    //ASSERT_FALSE(om.addAttr(om.getSourceOp(input), "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10)));

}

TEST(op_model, op_removal)
{

    mv::OpModel om("TestModel");

    auto input = om.input({32, 32, 1}, mv::DTypeType::Float16, mv::Order("CHW"));
    std::vector<double> weightsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});
    auto weights1 = om.constant(weightsData, {3, 3, 1, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto conv = om.conv(input, weights1, {4, 4}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    om.output(conv);
    
    mv::DataModel dm(om);
    unsigned tensorsCount = dm.tensorsCount();
    unsigned opsCount = om.opsCount();

    om.removeOp(convOp);
    ASSERT_FALSE(om.isValid(convOp));
    ASSERT_EQ(dm.tensorsCount(), tensorsCount - 1);
    ASSERT_EQ(om.opsCount(), opsCount - 1);

}
