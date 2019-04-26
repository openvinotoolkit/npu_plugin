#include "gtest/gtest.h"

int main(int argc, char **argv)
{
 //   ::testing::InitGoogleTest(&argc, argv);
//    ::testing::FLAGS_gtest_filter = "GlobalConfigParams*";
    //::testing::GTEST_FLAG(filter) = "insert_barrier_tasks*";
    ::testing::GTEST_FLAG(filter) = "combi/workloads_ztile_simple*";
    return RUN_ALL_TESTS();
}

