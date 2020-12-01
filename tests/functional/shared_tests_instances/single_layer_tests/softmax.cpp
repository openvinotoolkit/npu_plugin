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
        InferenceEngine::Precision inputPrecision;
        InferenceEngine::Precision outputPrecision;
        InferenceEngine::Layout inputLayout;
        InferenceEngine::Layout outputLayout;
        InferenceEngine::SizeVector inputShape;
        size_t axisInd;
        std::tie(std::ignore,
                 inputPrecision, outputPrecision,
                 inputLayout, outputLayout,
                 inputShape,
                 axisInd,
                 std::ignore,
                 std::ignore) = GetParam();

        if (!envConfig.IE_VPUX_USE_EXPERIMENTAL_COMPILER) {
            throw LayerTestsUtils::KmbSkipTestException("Blobs generated with MCM compiler hangs on runtime");
        }

        // TODO: [Track number: S#40296]
        if (inputShape.at(axisInd) == 1) {
            throw LayerTestsUtils::KmbSkipTestException("SoftMax over dim==1 fails during blob parsing");
        }

        if (envConfig.IE_VPUX_USE_EXPERIMENTAL_COMPILER) {
            if ((inputPrecision != InferenceEngine::Precision::FP16 && inputPrecision != InferenceEngine::Precision::UNSPECIFIED) ||
                (outputPrecision != InferenceEngine::Precision::FP16 && outputPrecision != InferenceEngine::Precision::UNSPECIFIED)) {
                throw LayerTestsUtils::KmbSkipTestException("Experimenal compiler supports only FP16");
            }

            if (inputShape.size() != 4) {
                throw LayerTestsUtils::KmbSkipTestException("Experimenal compiler supports only 4D");
            }

            if ((inputLayout != InferenceEngine::Layout::NCHW && inputLayout != InferenceEngine::Layout::ANY) ||
                (outputLayout != InferenceEngine::Layout::NCHW && outputLayout != InferenceEngine::Layout::ANY)) {
                throw LayerTestsUtils::KmbSkipTestException("Experimenal compiler supports only NCHW");
            }
        }
    }
};

TEST_P(KmbSoftMaxLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inputShapes2D = {
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
    testing::ValuesIn(inputLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    smoke_SoftMax2D,
    KmbSoftMaxLayerTest,
    params2D,
    SoftMaxLayerTest::getTestCaseName
);

const std::vector<InferenceEngine::SizeVector> inputShapes4D = {
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
    testing::ValuesIn(inputShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    smoke_SoftMax4D,
    KmbSoftMaxLayerTest,
    params4D,
    SoftMaxLayerTest::getTestCaseName
);

}  // namespace
