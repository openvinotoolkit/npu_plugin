#include <cmath>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/mcm/target/kmb/pwl/pwl_dyn_fit.hpp"

using namespace ::testing;

TEST(PWLFitBehavior, solve_if_empty_fit_params_result_in_exception) {
    PWLFloatFit pwl;
    ASSERT_ANY_THROW(pwl.solve());
}

TEST(PWLFitBehavior, solve_if_empty_range_result_in_exception) {
    PWLFloatFit pwl;
    pwl.setFunction(tanhf);
    pwl.setBitness(3);
    ASSERT_ANY_THROW(pwl.solve());
}

TEST(PWLFitBehavior, unsupported_bitness_exception) {
    PWLFloatFit pwl;
    ASSERT_ANY_THROW(pwl.setBitness(33));
    ASSERT_ANY_THROW(pwl.setBitness(0));
}

TEST(PWLFitBehavior, solve_if_empty_bitness_result_in_exception) {
    PWLFloatFit pwl;
    pwl.setFunction(tanhf);
    pwl.setRange(-5, 5);
    ASSERT_ANY_THROW(pwl.solve());
}

TEST(PWLFitBehavior, solve_if_empty_function_result_in_exception) {
    PWLFloatFit pwl;
    pwl.setRange(-5, 5);
    pwl.setBitness(3);
    ASSERT_ANY_THROW(pwl.solve());
}

TEST(PWLFitBehavior, second_solve_suceed) {
    PWLFloatFit pwl;
    pwl.setBitness(8);
    pwl.setRange(0, 5);
    pwl.setFunction(tanhf);

    ASSERT_NO_THROW(pwl.solve());

    pwl.setRange(0, 6);
    ASSERT_NO_THROW(pwl.solve());
}

TEST(PWLFitBehavior, setrange_if_invalid_range_result_in_exception) {
    PWLFloatFit pwl;
    ASSERT_ANY_THROW(pwl.setRange(-5, -6));

    //at least one float number should be in given range
    ASSERT_ANY_THROW(pwl.setRange(-5, -5 + std::numeric_limits<float>::epsilon()));
}

TEST(PWLFitBehavior, setinterval_if_invalid_number_of_intervals_result_in_exception) {
    PWLFloatFit pwl;
    ASSERT_ANY_THROW(pwl.setMaxIntervals(0));
}

TEST(PWLFitBehavior, solve_give_most_8_segments) {
    PWLFloatFit pwl;
    pwl.setRange(-5, 5);
    pwl.setFunction(tanhf);
    pwl.setBitness(8);
    pwl.setMaxIntervals(8);

    auto pwl_function = pwl.solve();

    //at least one float number should be in given range
    ASSERT_LE(pwl_function.segments().size(), 8);
    ASSERT_GE(pwl_function.segments().size(), 1);
}

TEST(PWLFitBehavior, solve_produces_function_defined_everywhere) {
    PWLFloatFit pwl;
    pwl.setRange(-5, 5);
    pwl.setFunction(tanhf);
    pwl.setBitness(8);
    pwl.setMaxIntervals(8);

    auto pwl_function = pwl.solve();

    //at least one float number should be in given range
    for (int i = -(1 << 7); i != (1 << 7); i++) {
        ASSERT_NO_THROW(pwl_function(i));
    }
}

TEST(PWLFitBehavior, solve_produces_function_unique_everywhere) {
    PWLFloatFit pwl;
    pwl.setRange(-5, 5);
    pwl.setFunction(tanhf);
    pwl.setBitness(8);
    pwl.setMaxIntervals(8);

    auto pwl_function = pwl.solve();

    //at least one float number should be in given range
    for (int i = -(1 << 7); i != (1 << 7); i++) {
        std::vector<int> contains;
        for (auto &&s : pwl_function.segments()) {
            if (s.contains(i)) {
                contains.push_back(i);
            }
        }
        ASSERT_THAT(contains, ElementsAre(i));
    }
}

TEST(PWLFitBehavior, PWLFunction_Segment_cost_cannot_evaluate_if_incorrect_bitness) {
    PWLFunction::Segment ss(0,0,0,0);
    std::vector<int> refFnc;
    ASSERT_ANY_THROW(ss.cost(refFnc, 0));
    ASSERT_ANY_THROW(ss.cost(refFnc, 33));
}

TEST(PWLFitBehavior, PWLFunction_Segment_cost_cannot_evaluate_if_incorrect_number_of_values_for_ref_function) {
    PWLFunction::Segment ss(0,0,0,0);
    std::vector<int> refFnc(255);
    ASSERT_ANY_THROW(ss.cost(refFnc, 8));

    std::vector<int> refFnc3(4095);
    ASSERT_ANY_THROW(ss.cost(refFnc3, 13));
}

TEST(PWLFitBehavior, PWLFunction_Segment_cost_cannot_evaluate_if_segment_out_of_interval) {
    std::vector<int> refFnc(256);

    ASSERT_ANY_THROW((PWLFunction::Segment{128,0,0,0}.cost(refFnc, 8)));
    ASSERT_ANY_THROW((PWLFunction::Segment{127,2,0,0}.cost(refFnc, 8)));
    ASSERT_ANY_THROW((PWLFunction::Segment{-129,0,0,0}.cost(refFnc, 8)));
    // point of 127 is included
    ASSERT_NO_THROW((PWLFunction::Segment{127,1,0,0}.cost(refFnc, 8)));
}

TEST(PWLFitBehavior, PWLFunction_Segment_fit_cannot_evaluate_if_incorrect_bitness) {
    PWLFunction::Segment ss(0,0,0,0);
    std::vector<int> refFnc;
    ASSERT_ANY_THROW(ss.fit(refFnc, 0, {0, 0}, {0, 0}));
    ASSERT_ANY_THROW(ss.fit(refFnc, 33, {0, 0}, {0, 0}));
}

TEST(PWLFitBehavior, PWLFunction_Segment_fit_cannot_evaluate_if_incorrect_number_of_values_for_ref_function) {
    PWLFunction::Segment ss(0,0,0,0);
    std::vector<int> refFnc(255);
    ASSERT_ANY_THROW(ss.fit(refFnc, 8, {0, 0}, {0, 0}));

    std::vector<int> refFnc3(4095);
    ASSERT_ANY_THROW(ss.fit(refFnc3, 13, {0, 0}, {0, 0}));
}

TEST(PWLFitBehavior, PWLFunction_Segment_fit_cannot_evaluate_if_segment_out_of_interval) {
    std::vector<int> refFnc(256);

    ASSERT_ANY_THROW((PWLFunction::Segment{128,1,0,0}.fit(refFnc, 8, {0, 0}, {0, 0})));
    ASSERT_ANY_THROW((PWLFunction::Segment{127,1,0,0}.fit(refFnc, 8, {0, 0}, {0, 0})));
    ASSERT_ANY_THROW((PWLFunction::Segment{-129,1,0,0}.fit(refFnc, 8, {0, 0}, {0, 0})));
    // invalid fit range of zero length
    ASSERT_ANY_THROW((PWLFunction::Segment{127,0,0,0}.fit(refFnc, 8, {0, 0}, {0, 0})));
}

TEST(PWLFitBehavior, PWLFunction_Segment_fit_cannot_evaluate_if_biases_range_is_empty) {
    std::vector<int> refFnc(256);

    ASSERT_ANY_THROW((PWLFunction::Segment{126,1,0,0}.fit(refFnc, 8, {0, 0}, {0, 1})));
    ASSERT_NO_THROW((PWLFunction::Segment{126,1,0,0}.fit(refFnc, 8, {0, 1}, {0, 1})));
}

TEST(PWLFitBehavior, PWLFunction_Segment_fit_cannot_evaluate_if_shift_range_is_empty) {
    std::vector<int> refFnc(256);

    ASSERT_ANY_THROW((PWLFunction::Segment{126,1,0,0}.fit(refFnc, 8, {0, 1}, {0, 0})));
    ASSERT_NO_THROW((PWLFunction::Segment{126,1,0,0}.fit(refFnc, 8, {0, 1}, {0, 1})));
}