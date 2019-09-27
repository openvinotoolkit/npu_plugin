#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(group, op_member_append)
{

    /*mv::OpModel om("TestModel");

    auto input = om.input({8, 8, 1}, mv::DType("Float16"), mv::Order("CHW"));
    auto inputOp = om.getSourceOp(input);
    auto weights = om.constant(mv::utils::generateSequence<double>(1), {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("CHW"));
    auto weightsOp = om.getSourceOp(weights);
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    auto convOp = om.getSourceOp(conv);
    auto pool = om.maxPool(input, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto poolOp = om.getSourceOp(pool);
    om.output(conv);
    auto outputOp = convOp.leftmostChild();

    std::string groupName = "group1";
    auto group1 = om.addGroup(groupName);

    om.addGroupElement(inputOp, group1);
    om.addGroupElement(weightsOp, group1);
    om.addGroupElement(convOp, group1);
    om.addGroupElement(outputOp, group1);

    for (auto it = inputOp; it != om.opEnd(); ++it)
    {
        if (it != poolOp)
        {
            ASSERT_TRUE(it->hasAttr("groups"));
            ASSERT_EQ(it->get<std::vector<std::string>>("groups").size(), 1);
            ASSERT_EQ(it->get<std::vector<std::string>>("groups")[0], groupName);
        }
    }
    
    ASSERT_EQ(group1->size(), 4);
    ASSERT_TRUE(group1->find(*inputOp) != group1->end());
    ASSERT_TRUE(group1->find(*weightsOp) != group1->end());
    ASSERT_TRUE(group1->find(*convOp) != group1->end());
    ASSERT_TRUE(group1->find(*outputOp) != group1->end());
    ASSERT_TRUE(group1->find(*poolOp) == group1->end());*/

}

TEST(group, op_member_remove)
{

    /*mv::OpModel om("TestModel");

    auto input = om.input({8, 8, 1}, mv::DType("Float16"), mv::Order("CHW"));
    auto inputOp = om.getSourceOp(input);
    auto weights = om.constant(mv::utils::generateSequence<double>(1), {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("CHW"));
    auto weightsOp = om.getSourceOp(weights);
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    auto convOp = om.getSourceOp(conv);
    om.output(conv);
    auto outputOp = convOp.leftmostChild();

    std::string groupName = "group1";
    auto group1 = om.addGroup(groupName);

    om.addGroupElement(inputOp, group1);
    om.addGroupElement(weightsOp, group1);
    om.addGroupElement(convOp, group1);
    om.addGroupElement(outputOp, group1);

    om.removeGroupElement(inputOp, group1);
    om.removeGroupElement(outputOp, group1);

    ASSERT_EQ(group1->size(), 2);
    ASSERT_TRUE(group1->find(*inputOp) == group1->end());
    ASSERT_TRUE(group1->find(*weightsOp) != group1->end());
    ASSERT_TRUE(group1->find(*convOp) != group1->end());
    ASSERT_TRUE(group1->find(*outputOp) == group1->end());

    ASSERT_FALSE(inputOp->hasAttr("groups"));
    ASSERT_FALSE(outputOp->hasAttr("groups"));*/

}