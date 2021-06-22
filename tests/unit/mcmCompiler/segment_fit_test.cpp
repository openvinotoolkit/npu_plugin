#include <algorithm>
#include <numeric>
#include "gtest/gtest.h"
#include "include/mcm/target/kmb/pwl/pwl_dyn_fit.hpp"

enum FitType {
    eFast,
    eBruteForce
};

using SegmentFitTestParams = std::tuple<FitType>;

class SegmentFitTests : public testing::TestWithParam<SegmentFitTestParams> {
protected:
    std::vector<int> refFunctionU8;
    std::function<int (const std::vector<int> & refFunction, int nBits) > fitCallback;
    PWLFunction::Segment ss{0, 10};

    PWLFunction::Range bias{-1024, 2048};
    PWLFunction::Range shift{-8, 16};

public:
    static std::string getTestCaseName(const testing::TestParamInfo<SegmentFitTestParams> & obj) {
        FitType fitType;
        std::tie(fitType) = obj.param;
        if (fitType == eFast) {
            return "fitFast";
        }
        return "fit";
    }
    void SetUp() override {
        FitType fitType;
        std::tie(fitType) = GetParam();
        if (fitType == eFast) {
            fitCallback = [this](const std::vector<int> & refFunction, int nBits) {
                return ss.fitfast(refFunction, nBits);
            };
        } else {
            fitCallback =  [this](const std::vector<int> & refFunction, int nBits) {
                return ss.fit(refFunction, nBits, bias, shift);
            };
        }

        int positionValues = std::numeric_limits<uint8_t>::max() + 1;
        refFunctionU8.resize(positionValues);
        iota(refFunctionU8.begin(), refFunctionU8.end(), std::numeric_limits<char>::min());
    }
};

TEST_P(SegmentFitTests, can_fit_y_eq_0_125_x_with_zero_accuracy_drop) {

    for (auto & v : refFunctionU8) {
        v *= 0.125;
    }
    ASSERT_EQ(0, fitCallback(refFunctionU8, 8));
    ASSERT_EQ(-3, ss.getShift());
    ASSERT_EQ(0, ss.getBias());
}

TEST_P(SegmentFitTests, can_fit_exact_mish_19_one_segment_with_1_accuracy_drop) {

    refFunctionU8[128 - 10] = -3;
    refFunctionU8[128 - 9]  = -3;
    refFunctionU8[128 - 8]  = -3;
    refFunctionU8[128 - 7]  = -3;
    refFunctionU8[128 - 6]  = -2;
    ss = {-10, 5};
    ASSERT_EQ(1, fitCallback(refFunctionU8, 8));
}


TEST_P(SegmentFitTests, can_fit_y_eq_x_with_zero_accuracy_drop) {

    ASSERT_EQ(0, fitCallback(refFunctionU8, 8));
    ASSERT_EQ(0, ss.getShift());
    ASSERT_EQ(0, ss.getBias());
}


TEST_P(SegmentFitTests, can_fit_y_eq_x_plus_1_with_zero_accuracy_drop) {

    for (auto & v : refFunctionU8) {
        v ++;
    }

    ASSERT_EQ(0, fitCallback(refFunctionU8, 8));
    ASSERT_EQ(0, ss.getShift());
    ASSERT_EQ(1, ss.getBias());
}

TEST_P(SegmentFitTests, can_fit_y_eq_minus_x_with_horizontal_segment_only) {

    for (auto & v : refFunctionU8) {
        v = -v;
    }

    ss = {0, 4};
    // optimal is line of y = -2 or y = -1 with error = 4
    ASSERT_EQ(2 + 1 + 1, fitCallback(refFunctionU8, 8));
    ASSERT_EQ(-8, ss.getShift());
    ASSERT_TRUE(-2 == ss.getBias() || -1 == ss.getBias());
}

INSTANTIATE_TEST_SUITE_P(SegmentFitTestsAll, SegmentFitTests,
        ::testing::Values(eFast, eBruteForce),
        SegmentFitTests::getTestCaseName);