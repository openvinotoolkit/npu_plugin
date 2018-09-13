#include "gtest/gtest.h"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <cstdlib>

TEST(target_descriptor, load_from_file)
{

    std::string descPath = mv::utils::projectRootPath() + std::string("/config/target/ma2480.json");
    mv::TargetDescriptor desc;
    ASSERT_TRUE(desc.load(descPath));
    ASSERT_EQ(desc.getTarget(), mv::Target::ma2480);

}  

TEST(target_descriptor, compose)
{

    mv::TargetDescriptor desc;
    
    desc.setTarget(mv::Target::ma2480);
    desc.setDType(mv::DTypeType::Float16);
    desc.setOrder(mv::OrderType::ColumnMajor);

    desc.appendAdaptPass("adaptPass1");
    desc.appendAdaptPass("adaptPass2", 0);

    desc.appendOptPass("optPass1");
    desc.appendFinalPass("finalPass1");
    desc.appendSerialPass("serialPass1");
    desc.appendValidPass("validPass1");

    desc.defineOp(mv::OpType::Conv2D);

    ASSERT_EQ(desc.getTarget(), mv::Target::ma2480);
    ASSERT_EQ(desc.getDType(), mv::DTypeType::Float16);
    ASSERT_EQ(desc.getOrder(), mv::OrderType::ColumnMajor);

    ASSERT_EQ(desc.adaptPassesCount(), 2);
    ASSERT_EQ(desc.optPassesCount(), 1);
    ASSERT_EQ(desc.finalPassesCount(), 1);
    ASSERT_EQ(desc.serialPassesCount(), 1);
    ASSERT_EQ(desc.validPassesCount(), 1);

    ASSERT_EQ(desc.adaptPasses()[0], "adaptPass2");
    ASSERT_EQ(desc.adaptPasses()[1], "adaptPass1");
    ASSERT_TRUE(desc.opSupported(mv::OpType::Conv2D));
    ASSERT_FALSE(desc.opSupported(mv::OpType::MaxPool2D));

}

