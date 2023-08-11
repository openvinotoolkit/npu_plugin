//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/eltwise.hpp"
#include <vector>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "vpux_layer_test.hpp"

namespace ov::test::subgraph {

class VPUXEltwiseLayerTest : public EltwiseLayerTest, virtual public VPUXLayerTest {};

using namespace ngraph::helpers;

TEST_P(VPUXEltwiseLayerTest, VPU3700_SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        if (eltwiseType == EltwiseTypes::MOD || eltwiseType == EltwiseTypes::ERF) {
            skip << "Type is not supported";
        }
        if (eltwiseType == EltwiseTypes::SQUARED_DIFF || eltwiseType == EltwiseTypes::SUBTRACT) {
            skip << "Unsupported type in SW mode";
        }
    });

    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(VPUXEltwiseLayerTest, VPU3700_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        if (eltwiseType == EltwiseTypes::MOD || eltwiseType == EltwiseTypes::ERF) {
            skip << "Type is not supported";
        }
        if (eltwiseType == EltwiseTypes::SQUARED_DIFF) {
            skip << "Squared difference not supported in HW mode";
        }
    });

    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(VPUXEltwiseLayerTest, VPU3720_SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        // [Tracking number: E#73421]
        if (eltwiseType == EltwiseTypes::SQUARED_DIFF || eltwiseType == EltwiseTypes::ERF) {
            skip << "Not compiling";
        }
        if (eltwiseType == EltwiseTypes::MOD && netPrecisions == ov::element::i32) {
            skip << "Not implemented";
        }
    });

    setSkipInferenceCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        // [Tracking number: E#73421]
        if (eltwiseType == EltwiseTypes::MOD) {
            skip << "Type is not supported";
        }
    });

    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(VPUXEltwiseLayerTest, VPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        // [Tracking number: E#73421]
        if (eltwiseType == EltwiseTypes::SQUARED_DIFF || eltwiseType == EltwiseTypes::ERF) {
            skip << "Not compiling";
        }
        // [Tracking number: E#82236]
        if (netPrecisions == ov::element::i32) {
            skip << "Type is not supported";
        }
    });

    setSkipInferenceCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto secondInputType = std::get<2>(GetParam());

        // [Tracking number: E#73421]
        if (eltwiseType == EltwiseTypes::MOD) {
            skip << "Type is not supported";
        }
    });

    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace ov::test::subgraph

namespace {

using namespace ov::test::subgraph;

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f16,
        ov::element::i32,
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::VECTOR,
        CommonTestUtils::OpType::SCALAR,
};

//
// Test supported Eltwise types + Tiling
//

std::set<EltwiseTypes> eltwiseTypes = {EltwiseTypes::ADD,       EltwiseTypes::MULTIPLY,     EltwiseTypes::SUBTRACT,
                                       EltwiseTypes::DIVIDE,    EltwiseTypes::SQUARED_DIFF, EltwiseTypes::POWER,
                                       EltwiseTypes::FLOOR_MOD, EltwiseTypes::MOD,          EltwiseTypes::ERF};

std::vector<std::vector<ov::Shape>> bigShape = {{{1, 10, 256, 256}, {1, 10, 256, 256}}};

const auto typesParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypes, VPUXEltwiseLayerTest, typesParams,
                         VPUXEltwiseLayerTest::getTestCaseName);

//
// Scalar mode
//

std::vector<std::vector<ov::Shape>> inShapesScalar = {
        {{10}},              // 1D
        {{1, 9}},            // NC
        {{1, 128, 32}},      // CHW
        {{1, 3, 224, 224}},  // NCHW
};

const auto scalarParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(CommonTestUtils::OpType::SCALAR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesND, VPUXEltwiseLayerTest, scalarParams,
                         VPUXEltwiseLayerTest::getTestCaseName);

//
// Vector mode
//

std::vector<std::vector<ov::Shape>> inShapesVector = {
        {{24}, {24}},                          // 1D
        {{1, 9}, {1, 1}},                      // NC + scalar
        {{1, 128, 32}, {1, 128, 32}},          // CHW, eltwise
        {{1, 128, 32}, {1, 128, 1}},           // CHW, input1 != input2, broadcast over W
        {{1, 128, 32}, {1, 1, 32}},            // CHW, input1 != input2, broadcast over H
        {{1, 128, 32}, {1, 1, 1}},             // CHW + scalar
        {{1, 3, 224, 224}, {1, 3, 224, 224}},  // NCHW, eltwise
        {{1, 3, 224, 224}, {1, 1, 1, 1}},      // NCHW + scalar
        {{1, 3, 224, 224}, {1, 3, 1, 1}},      // NCHW, broadcast over HW
        {{2, 3, 224, 224}, {1, 1, 1, 224}},    // NCHW, N != 1, broadcast over NCH
};

const auto vectorParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesND, VPUXEltwiseLayerTest, vectorParams,
                         VPUXEltwiseLayerTest::getTestCaseName);

}  // namespace
