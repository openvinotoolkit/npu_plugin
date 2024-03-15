//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/eltwise.hpp"
#include <vector>
#include "ov_models/utils/ov_helpers.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class EltwiseLayerTestCommon : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

class EltwiseLayerF32TestCommon : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

class EltwiseEmptyShapeInputTest : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

using namespace ngraph::helpers;

TEST_P(EltwiseLayerTestCommon, NPU3700_SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        if (eltwiseType == EltwiseTypes::MOD) {
            skip << "Type is not supported";
        }
        if (eltwiseType == EltwiseTypes::SQUARED_DIFF || eltwiseType == EltwiseTypes::SUBTRACT) {
            skip << "Unsupported type in SW mode";
        }
    });

    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(EltwiseLayerTestCommon, NPU3700_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        if (eltwiseType == EltwiseTypes::MOD) {
            skip << "Type is not supported";
        }
        if (eltwiseType == EltwiseTypes::SQUARED_DIFF) {
            skip << "Squared difference not supported in HW mode";
        }
    });

    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(EltwiseLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(EltwiseLayerTestCommon, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        // [Tracking number: E#82236]
        if (netPrecisions == ov::element::i32) {
            skip << "Type is not supported";
        }
    });

    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(EltwiseLayerF32TestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(EltwiseLayerF32TestCommon, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        if (netPrecisions == ov::element::f32) {
            skip << "FP32 operations will be converted to IE.scaleshift in AdjustScaleShiftForDWConv in HW Mode. "
                    "IE.scaleshift is a NCE task which do not support FP32";
        }
    });

    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(EltwiseEmptyShapeInputTest, NPU3720_HW) {
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

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::VECTOR,
        ov::test::utils::OpType::SCALAR,
};

//
// Test supported Eltwise types + Tiling
//

std::set<EltwiseTypes> eltwiseTypes = {EltwiseTypes::ADD,       EltwiseTypes::MULTIPLY,     EltwiseTypes::SUBTRACT,
                                       EltwiseTypes::DIVIDE,    EltwiseTypes::SQUARED_DIFF, EltwiseTypes::POWER,
                                       EltwiseTypes::FLOOR_MOD, EltwiseTypes::MOD};

std::set<EltwiseTypes> eltwiseTypesF32 = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::POWER,
                                          EltwiseTypes::DIVIDE};

std::vector<std::vector<ov::Shape>> bigShape = {{{1, 10, 256, 256}, {1, 10, 256, 256}}};

const auto typesParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypes, EltwiseLayerTestCommon, typesParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto typesParamsF32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
        ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes), ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisionsF32), ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypesF32, EltwiseLayerF32TestCommon, typesParamsF32,
                         EltwiseLayerF32TestCommon::getTestCaseName);

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
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesND, EltwiseLayerTestCommon, scalarParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto scalarParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesNDF32, EltwiseLayerF32TestCommon, scalarParamsF32,
                         EltwiseLayerF32TestCommon::getTestCaseName);

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
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesND, EltwiseLayerTestCommon, vectorParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto vectorParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesNDF32, EltwiseLayerF32TestCommon, vectorParamsF32,
                         EltwiseLayerF32TestCommon::getTestCaseName);

//
//  This case to test the support for empty shape input for Add and Multiply ops
//
std::set<EltwiseTypes> eltwise0DInputOps = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY};

std::vector<std::vector<ov::Shape>> eltwise0DInputShape = {
        {{}},  // 0D
};

const auto vectorParamsEmptyShapeInput =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(eltwise0DInputShape)),
                           ::testing::ValuesIn(eltwise0DInputOps), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_0DInputTest, EltwiseEmptyShapeInputTest, vectorParamsEmptyShapeInput,
                         EltwiseEmptyShapeInputTest::getTestCaseName);

}  // namespace
