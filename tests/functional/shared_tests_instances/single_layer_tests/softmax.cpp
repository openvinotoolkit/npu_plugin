// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/softmax.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSoftMaxLayerTest: public SoftMaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        InferenceEngine::Precision inPrecision;
        InferenceEngine::Precision outPrecision;
        InferenceEngine::SizeVector inShape;
        size_t axisInd;
        std::tie(std::ignore,
                 inPrecision, outPrecision,
                 std::ignore, std::ignore,
                 inShape,
                 axisInd,
                 std::ignore,
                 std::ignore) = GetParam();

        if (isCompilerMCM()) {
            // [Track number: S#44702]
            if (inPrecision == InferenceEngine::Precision::FP32 || outPrecision == InferenceEngine::Precision::FP32) {
                throw LayerTestsUtils::KmbSkipTestException("SoftMax with FP32 input/output hangs on graph loading");
            }

            // [Track number: S#40296]
            if (inShape.at(axisInd) == 1) {
                throw LayerTestsUtils::KmbSkipTestException("SoftMax over dim==1 fails during blob parsing");
            }
        }
    }

    void SkipBeforeValidate() override {
        InferenceEngine::Precision inPrecision;
        std::tie(std::ignore,
                 inPrecision, std::ignore,
                 std::ignore, std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore) = GetParam();

        // [Track number: S#44702]
        if (inPrecision == InferenceEngine::Precision::U8) {
            throw LayerTestsUtils::KmbSkipTestException("SoftMax with U8 input produces wrong results");
        }
    }
};

TEST_P(KmbSoftMaxLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbSoftMaxLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inShapes2D = {
    InferenceEngine::SizeVector {1, 100},
    InferenceEngine::SizeVector {100, 1},
    InferenceEngine::SizeVector {10, 10},
};

const std::vector<size_t> axis2D = {
    0, 1
};

const auto params2D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::ValuesIn(inLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
    smoke_SoftMax2D,
    KmbSoftMaxLayerTest,
    params2D,
    SoftMaxLayerTest::getTestCaseName
);

const std::vector<InferenceEngine::SizeVector> inShapes4D = {
    InferenceEngine::SizeVector {1, 100, 1, 1},
    InferenceEngine::SizeVector {1, 3, 4, 3},
    InferenceEngine::SizeVector {2, 3, 4, 5},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
    smoke_SoftMax4D,
    KmbSoftMaxLayerTest,
    params4D,
    SoftMaxLayerTest::getTestCaseName
);

// internal tests

const std::vector<InferenceEngine::Precision> inPrecisionsInt = {
    InferenceEngine::Precision::U8,
    // TODO: nGraph testing framework doesn't support FP16
//    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> outPrecisionsInt = {
    // TODO: nGraph testing framework doesn't support FP16
//    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const auto params2DInt = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::ValuesIn(inPrecisionsInt),
    testing::ValuesIn(outPrecisionsInt),
    testing::ValuesIn(inLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
    internal_SoftMax2D,
    KmbSoftMaxLayerTest,
    params2DInt,
    SoftMaxLayerTest::getTestCaseName
);

const std::vector<InferenceEngine::SizeVector> inShapes4DInt = {
    InferenceEngine::SizeVector {1, 3, 32, 32},
    InferenceEngine::SizeVector {1, 3, 16, 16},
};

const std::vector<InferenceEngine::Layout> layouts4DInt = {
    InferenceEngine::Layout::NCHW,
    InferenceEngine::Layout::NHWC,
};

const auto params4DInt = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::ValuesIn(inPrecisionsInt),
    testing::ValuesIn(outPrecisionsInt),
    testing::ValuesIn(layouts4DInt),
    testing::ValuesIn(layouts4DInt),
    testing::ValuesIn(inShapes4DInt),
    testing::ValuesIn(axis4D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
    internal_SoftMax4D,
    KmbSoftMaxLayerTest,
    params4DInt,
    SoftMaxLayerTest::getTestCaseName
);

}  // namespace
