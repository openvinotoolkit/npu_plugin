// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/eltwise.hpp"
#include <vector>
#include "vpux_layer_test.hpp"

namespace ov {
namespace test {
namespace subgraph {

using namespace VPUXLayerTestsUtils;

class VPUXEltwiseLayerTest : public EltwiseLayerTest, virtual public VPUXLayerTestsCommon {};

class VPUXEltwiseLayerTest_MCM : public VPUXEltwiseLayerTest {
    SkipMessage SkipBeforeValidate() override {
        std::vector<InputShape> inShapes;
        std::tie(inShapes, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore) = GetParam();

        std::set<std::vector<ngraph::Shape>> badShapes = {{{2, 200}},
                                                          {{10, 200}},
                                                          {{1, 4, 4, 1}},
                                                          {{2, 17, 5, 4}, {1, 17, 1, 1}},
                                                          {{2, 17, 5, 1}, {1, 17, 1, 4}}};

        // [Track number: S#51346]
        for (const auto& inShape : inShapes) {
            if (badShapes.count(inShape.second)) {
                return {"Mismatch in comparison"};
            }
        }

        return vpux::None;
    }
};
class VPUXEltwiseLayerTest_MLIR : public VPUXEltwiseLayerTest {};

//
//[Track number: E#15146]
//
TEST_P(VPUXEltwiseLayerTest_MCM, DISABLED_MCM) {
    run();
}

//
//[Track number: E#30253]
//
TEST_P(VPUXEltwiseLayerTest_MLIR, MLIR_SW) {
    abs_threshold = 0.005;
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    run();
}

TEST_P(VPUXEltwiseLayerTest_MLIR, DISABLED_MLIR_HW) {
    abs_threshold = 0.005;
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    run();
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov

using namespace ov::test::subgraph;

namespace {

std::vector<std::vector<ov::Shape>> inShapes = {
        {{2}},
        {{2, 200}},
        {{10, 200}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{4, 4, 16}},
        {{1, 10, 100}},
        {{1, 4, 1, 1}},
        {{1, 1, 1, 3}},
        {{1, 4, 4, 1}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
};

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f16,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::CONSTANT,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::VECTOR,
        CommonTestUtils::OpType::SCALAR,
};

//
// MCM Instantiation
//

std::set<ngraph::helpers::EltwiseTypes> supportedTypesMCM{
        ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT, ngraph::helpers::EltwiseTypes::SQUARED_DIFF};

const auto eltwise_params_mcm =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                           ::testing::ValuesIn(supportedTypesMCM), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(CommonTestUtils::OpType::SCALAR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(testPlatformTargetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, VPUXEltwiseLayerTest_MCM, eltwise_params_mcm,
                         VPUXEltwiseLayerTest::getTestCaseName);

const auto eltwise_params_vector_mcm =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                           ::testing::ValuesIn(supportedTypesMCM), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(testPlatformTargetDevice), ::testing::Values(ov::test::Config{}));

//[Track number: S#51349]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_CompareWithRefs, VPUXEltwiseLayerTest_MCM, eltwise_params_vector_mcm,
                         VPUXEltwiseLayerTest::getTestCaseName);

//
// MLIR Instantiation
//

std::set<ngraph::helpers::EltwiseTypes> supportedTypesMLIR{
        ngraph::helpers::EltwiseTypes::DIVIDE, ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        ngraph::helpers::EltwiseTypes::POWER, ngraph::helpers::EltwiseTypes::FLOOR_MOD};

const auto eltwise_params_mlir =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                           ::testing::ValuesIn(supportedTypesMLIR), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(testPlatformTargetDevice), ::testing::Values(ov::test::Config{}));

// [Track number: E#15146]
// Initialization disabled partly
// INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, VPUXEltwiseLayerTest_MLIR, eltwise_params_mlir,
//                        VPUXEltwiseLayerTest::getTestCaseName);

// Specific add and multiply case

std::vector<std::vector<ov::Shape>> inSpecificShapes = {
        {{1, 9}},                          // NC
        {{1, 128, 32}},                    // CHW
        {{1, 128, 32}, {1, 128, 1}},       // CHW, input1 != input2, broadcast over W
        {{1, 128, 32}, {1, 1, 32}},        // CHW, input1 != input2, broadcast over H
        {{1, 9}, {1, 1}},                  // NC + scalar
        {{1, 128, 32}, {1, 1, 1}},         // CHW + scalar
        {{1, 3, 224, 224}, {1, 1, 1, 1}},  // NCHW, broadcast over HW + channels
        {{1, 3, 224, 224}, {1, 3, 1, 1}}};

const auto multiply_params_mlir = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inSpecificShapes)),
        ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::Values(ov::element::f16), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined), ::testing::Values(testPlatformTargetDevice),
        ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Multiply, VPUXEltwiseLayerTest_MLIR, multiply_params_mlir,
                         VPUXEltwiseLayerTest::getTestCaseName);

//===================================================================================================================
const auto power_params_mlir = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inSpecificShapes)),
        ::testing::Values(ngraph::helpers::EltwiseTypes::POWER),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::Values(ov::element::f16),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(testPlatformTargetDevice),
        ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Power, VPUXEltwiseLayerTest_MLIR, power_params_mlir,
                         VPUXEltwiseLayerTest::getTestCaseName);
//===================================================================================================================

const auto eltwise_add_params_mlir = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inSpecificShapes)),
        ::testing::Values(ngraph::helpers::EltwiseTypes::ADD), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::Values(ov::element::f16), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined), ::testing::Values(testPlatformTargetDevice),
        ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Add, VPUXEltwiseLayerTest_MLIR, eltwise_add_params_mlir,
                        VPUXEltwiseLayerTest::getTestCaseName);

// Specific subtract case

std::vector<std::vector<ov::Shape>> inSpecificSubtractShapes = {
        {{1, 2, 4}},
        {{1, 2, 2, 4}, {1, 2, 1, 1}},
};

const auto subtract_params_mlir = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inSpecificSubtractShapes)),
        ::testing::Values(ngraph::helpers::EltwiseTypes::SUBTRACT), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined), ::testing::Values(testPlatformTargetDevice),
        ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Specific_subtract, VPUXEltwiseLayerTest_MLIR, subtract_params_mlir,
                        VPUXEltwiseLayerTest::getTestCaseName);

}  // namespace
