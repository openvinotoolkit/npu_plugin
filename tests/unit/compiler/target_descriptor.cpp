#include "gtest/gtest.h"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <cstdlib>

TEST(target_descriptor, load_from_file)
{

    std::string descPath = mv::utils::projectRootPath() + std::string("/config/target/ma2490.json");
    mv::TargetDescriptor desc;
    ASSERT_TRUE(desc.load(descPath));
    ASSERT_EQ(desc.getTarget(), mv::Target::ma2490);

}  

class TargetDescriptorTest : public testing::TestWithParam<mv::Target> {};

TEST_Q(TargetDescriptorTest, compose)
{

    mv::TargetDescriptor desc;
    
    desc.setTarget(GetParam());
    desc.setDType(mv::DType("Float16"));

    desc.defineOp("Conv");

    ASSERT_EQ(desc.getTarget(), GetParam());
    ASSERT_EQ(desc.getDType(), mv::DType("Float16"));
    ASSERT_TRUE(desc.opSupported("Conv"));
    ASSERT_FALSE(desc.opSupported("UndefinedOp"));

}

INSTANTIATE_TEST_SUITE_P(Targets,
                         TargetDescriptorTest,
                         testing::Values(mv::Target::ma2490, mv::Target::ma3100));
