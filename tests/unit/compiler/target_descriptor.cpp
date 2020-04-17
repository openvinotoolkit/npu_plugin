#include "gtest/gtest.h"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <cstdlib>

class TargetDescriptorTest : public testing::TestWithParam<std::tuple<mv::Target, const char*>> {};

TEST_P(TargetDescriptorTest, load_from_file)
{

    std::string descPath = mv::utils::projectRootPath() + std::string("/config/target/") + std::get<1>(GetParam());
    mv::TargetDescriptor desc;
    ASSERT_TRUE(desc.load(descPath));
    ASSERT_EQ(desc.getTarget(), std::get<0>(GetParam()));

}

TEST_P(TargetDescriptorTest, compose)
{

    mv::TargetDescriptor desc;
    
    desc.setTarget(std::get<0>(GetParam()));
    desc.setDType(mv::DType("Float16"));

    desc.defineOp("Conv");

    ASSERT_EQ(desc.getTarget(), std::get<0>(GetParam()));
    ASSERT_EQ(desc.getDType(), mv::DType("Float16"));
    ASSERT_TRUE(desc.opSupported("Conv"));
    ASSERT_FALSE(desc.opSupported("UndefinedOp"));

}

INSTANTIATE_TEST_SUITE_P(Targets,
                         TargetDescriptorTest,
                         testing::Values(std::make_tuple(mv::Target::ma2490, "release_kmb.json"),
                                         std::make_tuple(mv::Target::ma3100, "release_thb.json")));
