// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/softmax.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSoftMaxLayerTest: public SoftMaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        InferenceEngine::Precision inPrecision;
        InferenceEngine::Precision outPrecision;
        std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>> inShapes;
        size_t axisInd;
        std::tie(std::ignore,
                 inPrecision, outPrecision,
                 std::ignore, std::ignore,
                 inShapes,
                 axisInd,
                 std::ignore,
                 std::ignore) = GetParam();

        if (isCompilerMCM()) {
            // [Track number: S#44702]
            if (inPrecision == InferenceEngine::Precision::FP32 || outPrecision == InferenceEngine::Precision::FP32) {
                throw LayerTestsUtils::KmbSkipTestException("SoftMax with FP32 input/output hangs on graph loading");
            }

            // [Track number: S#40296]
            for (const auto& shape : inShapes.second) {
                if (shape.at(axisInd) == 1) {
                    throw LayerTestsUtils::KmbSkipTestException("SoftMax over dim==1 fails during blob parsing");
                }
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

//         [Track number: S#44702]
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

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inShapes2D = {
        {{}, {{1, 100}}},
        {{}, {{100, 1}}},
        {{}, {{10, 10}}},
};

const std::vector<size_t> axis2D = {
        0, 1
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::ValuesIn(inLayouts2D),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inShapes2D),
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

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inShapes4D = {
        {{}, {{1, 2, 204, 62}}},
        {{}, {{1, 12, 2, 1444}}},
        {{}, {{1, 2, 72, 10}}},
        {{}, {{1, 4, 1, 1}}},
        {{}, {{1, 1000, 1, 1}}},
        {{}, {{300, 21, 1, 1}}}
};

const std::vector<InferenceEngine::Layout> layouts4D = {
        InferenceEngine::Layout::NCHW,
        InferenceEngine::Layout::NHWC,
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::ValuesIn(layouts4D),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inShapes4D),
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