#include "gtest/gtest.h"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <cstdlib>

namespace {

std::string getDescPath(mv::Target target)
{
    const char* filename;

    switch (target)
    {
        case mv::Target::ma2490:
            filename = "release_kmb.json";
            break;

        case mv::Target::ma3100:
            filename = "release_thb.json";
            break;

        default:
            ADD_FAILURE() << "Unimplemented target descriptor path";
            filename = "unimplemented";
            break;
    }

    return mv::utils::projectRootPath() + "/config/target/" + filename;
}

class TargetDescriptorTest : public testing::TestWithParam<mv::Target> {};

TEST_P(TargetDescriptorTest, load_from_file)
{

    mv::TargetDescriptor desc;
    ASSERT_TRUE(desc.load(getDescPath(GetParam())));
    ASSERT_EQ(desc.getTarget(), GetParam());

}

TEST_P(TargetDescriptorTest, compose)
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

TEST(ThbDmaControllerCount, load_from_file)
{

    mv::TargetDescriptor desc;
    ASSERT_TRUE(desc.load(getDescPath(mv::Target::ma3100)));
    ASSERT_EQ(desc.nceDefs().at("DMAControllers").totalNumber, 2);

}


}  // namespace
