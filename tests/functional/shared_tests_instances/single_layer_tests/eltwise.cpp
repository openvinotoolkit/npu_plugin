//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/eltwise.hpp"
#include <vector>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class VPUXEltwiseLayerTest : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

class VPUXEltwiseLayerF32Test : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

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
        if (eltwiseType == EltwiseTypes::ERF) {
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

TEST_P(VPUXEltwiseLayerF32Test, VPU3720_SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(VPUXEltwiseLayerF32Test, VPU3720_HW) {
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

        if (netPrecisions == ov::element::f32 &
            (eltwiseType == EltwiseTypes::ADD || eltwiseType == EltwiseTypes::MULTIPLY ||
             eltwiseType == EltwiseTypes::POWER)) {
            skip << "ADD will be converted to IE.scaleshift in AdjustScaleShiftForDWConv in HW Mode. "
                    "IE.scaleshift is a NCE task which do not support FP32";
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

std::vector<ov::test::ElementType> netPrecisionsF32 = {
        ov::element::f32,
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

std::set<EltwiseTypes> eltwiseTypesF32 = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::POWER};

std::vector<std::vector<ov::Shape>> bigShape = {{{1, 10, 256, 256}, {1, 10, 256, 256}}};

const auto typesParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypes, VPUXEltwiseLayerTest, typesParams,
                         VPUXEltwiseLayerTest::getTestCaseName);

const auto typesParamsF32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
        ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes), ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisionsF32), ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
        ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypesF32, VPUXEltwiseLayerF32Test, typesParamsF32,
                         VPUXEltwiseLayerF32Test::getTestCaseName);

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

const auto scalarParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(CommonTestUtils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesNDF32, VPUXEltwiseLayerF32Test, scalarParamsF32,
                         VPUXEltwiseLayerF32Test::getTestCaseName);

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

const auto vectorParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesNDF32, VPUXEltwiseLayerF32Test, vectorParamsF32,
                         VPUXEltwiseLayerF32Test::getTestCaseName);

}  // namespace
