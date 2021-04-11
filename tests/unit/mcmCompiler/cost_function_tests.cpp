#include <algorithm>
#include <numeric>
#include "gtest/gtest.h"
#include "include/mcm/target/kmb/pwl/pwl_dyn_fit.hpp"

class PWLFitCostFunctionTests : public ::testing::Test {
protected:
    std::vector<int> refFunctionU8;

public:
    void SetUp() override {
        int positionValues = std::numeric_limits<uint8_t>::max() + 1;
        refFunctionU8.resize(positionValues);
        iota(refFunctionU8.begin(), refFunctionU8.end(), std::numeric_limits<char>::min());
    }
};

TEST_F(PWLFitCostFunctionTests, on_a_single_segment) {
    // y = x
    PWLFunction::Segment ss{0, 10, 0, 0};
    ASSERT_EQ(0, ss.cost(refFunctionU8, 8));
}

TEST_F(PWLFitCostFunctionTests, on_a_single_segment_shift) {
    // y = x
    PWLFunction::Segment ss{0, 10, 1, 0};
    ASSERT_EQ(10, ss.cost(refFunctionU8, 8));
}

TEST_F(PWLFitCostFunctionTests, on_a_single_segment_slope) {
    // y = x
    PWLFunction::Segment ss{0, 4, 0, -8};
    ASSERT_EQ(1 + 2 + 3, ss.cost(refFunctionU8, 8));
}

TEST_F(PWLFitCostFunctionTests, on_a_single_segment_slope_for_negatives) {
    // y = x
    PWLFunction::Segment ss{-4, 4, 0, -8};
    ASSERT_EQ(3 + 2 + 1, ss.cost(refFunctionU8, 8));
}

TEST_F(PWLFitCostFunctionTests, on_a_single_segment_absolute_of_cost) {
    // y = x
    PWLFunction::Segment ss{0, 10, -1, 0};
    ASSERT_EQ(10, ss.cost(refFunctionU8, 8));
}
