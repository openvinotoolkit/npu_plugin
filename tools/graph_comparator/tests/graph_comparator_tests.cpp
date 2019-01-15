#include "gtest/gtest.h"
#include "tools/graph_comparator/include/graph_comparator/graph_comparator.hpp"
#include "include/mcm/utils/env_loader.hpp"

TEST(graph_comparator, load_valid_path)
{

    mv::tools::GraphComparator gc;
    std::string graph1Path = mv::utils::projectRootPath() + "/tools/graph_comparator/tests/data/graphfile_1";
    ASSERT_NO_THROW(gc.compare(graph1Path, graph1Path));
    
}

TEST(graph_comparator, load_invalid_path)
{

    mv::tools::GraphComparator gc;
    std::string graph1Path = "invalid_path";
    ASSERT_ANY_THROW(gc.compare(graph1Path, graph1Path));
    
}

TEST(graph_comparator, load_invalid_file)
{

    mv::tools::GraphComparator gc;
    std::string graph1Path = mv::utils::projectRootPath() + "/tools/graph_comparator/tests/data/graphfile_invalid";
    ASSERT_ANY_THROW(gc.compare(graph1Path, graph1Path));
    
}

TEST(graph_comparator, compare_same)
{

    mv::tools::GraphComparator gc;
    std::string graph1Path = mv::utils::projectRootPath() + "/tools/graph_comparator/tests/data/graphfile_1";
    ASSERT_TRUE(gc.compare(graph1Path, graph1Path));
    ASSERT_TRUE(gc.lastDiff().empty());
    
}

TEST(graph_comparator, compare_different)
{

    mv::tools::GraphComparator gc;
    std::string graph1Path = mv::utils::projectRootPath() + "/tools/graph_comparator/tests/data/graphfile_1";
    std::string graph2Path = mv::utils::projectRootPath() + "/tools/graph_comparator/tests/data/graphfile_2";
    ASSERT_FALSE(gc.compare(graph1Path, graph2Path));
    ASSERT_FALSE(gc.lastDiff().empty());
    
}


int main(int argc, char **argv)
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Silent);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
