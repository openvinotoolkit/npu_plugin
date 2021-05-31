// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/pooling.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPoolingLayerTest: public PoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMLIR()) {
            const auto& poolParams = std::get<0>(GetParam());
            auto poolType = ngraph::helpers::PoolingTypes{};
            auto strides = std::vector<size_t>{};
            auto padType = ngraph::op::PadType{};
            std::tie(poolType, std::ignore, strides, std::ignore, std::ignore, std::ignore, padType, std::ignore) =
                    poolParams;

            // MLIR uses software layer, which seem to be flawed
            // MCM uses hardware implementation of AvgPool, replacing with DW Conv
            if (poolType == ngraph::helpers::PoolingTypes::AVG &&
                padType == ngraph::op::PadType::VALID &&
                (strides.at(0) != 1 || strides.at(1) != 1)) {
                throw LayerTestsUtils::KmbSkipTestException("AVG pool with VALID PadType and strides != 1 produces inaccurate results");
            }
        }
    }
};

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MCM) {
    Run();
}

// [Track number: S#49089]
TEST_P(KmbPoolingLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

const std::vector<InferenceEngine::SizeVector> inShapes = {
    {1, 16, 30, 30}
};

////* ========== Max Polling ========== */

/* +========== Explicit Pad Floor Rounding ========== */

const std::vector<poolSpecificParams> maxPoolExplicitPadFloorRoundingParams = {
    std::make_tuple(
        PoolingTypes::MAX,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {1, 1},  // strides
        std::vector<size_t> {0, 0},  // padBegins
        std::vector<size_t> {0, 0},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::MAX,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {1, 1},  // padBegins
        std::vector<size_t> {1, 1},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::MAX,
        std::vector<size_t> {5, 5},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {2, 0},  // padBegins
        std::vector<size_t> {2, 0},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),
};

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_ExplicitPad_FloorRounding, KmbPoolingLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(maxPoolExplicitPadFloorRoundingParams),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inShapes),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */

const std::vector<poolSpecificParams> maxPoolExplicitPadCeilRoundingParams = {
    std::make_tuple(
        PoolingTypes::MAX,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {1, 1},  // strides
        std::vector<size_t> {0, 0},  // padBegins
        std::vector<size_t> {0, 0},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::MAX,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {1, 1},  // padBegins
        std::vector<size_t> {2, 2},  // padEnds
        ngraph::op::RoundingType::CEIL,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::MAX,
        std::vector<size_t> {5, 5},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {2, 2},  // padBegins
        std::vector<size_t> {3, 3},  // padEnds
        ngraph::op::RoundingType::CEIL,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),
};

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_ExplicitPad_CeilRounding, KmbPoolingLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(maxPoolExplicitPadCeilRoundingParams),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg Pooling ========== */

/* +========== Explicit Pad Floor Rounding ========== */

const std::vector<poolSpecificParams> avgPoolExplicitPadFloorRoundingParams = {
    std::make_tuple(
        PoolingTypes::AVG,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {1, 1},  // strides
        std::vector<size_t> {0, 0},  // padBegins
        std::vector<size_t> {0, 0},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::AVG,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {1, 1},  // padBegins
        std::vector<size_t> {1, 1},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::AVG,
        std::vector<size_t> {5, 5},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {2, 0},  // padBegins
        std::vector<size_t> {2, 0},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),
};

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_ExplicitPad_FloorRounding, KmbPoolingLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(avgPoolExplicitPadFloorRoundingParams),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Ceil Rounding ========== */

const std::vector<poolSpecificParams> avgPoolExplicitPadCeilRoundingParams = {
    std::make_tuple(
        PoolingTypes::AVG,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {1, 1},  // strides
        std::vector<size_t> {0, 0},  // padBegins
        std::vector<size_t> {0, 0},  // padEnds
        ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::AVG,
        std::vector<size_t> {3, 3},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {1, 1},  // padBegins
        std::vector<size_t> {2, 2},  // padEnds
        ngraph::op::RoundingType::CEIL,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),

    std::make_tuple(
        PoolingTypes::AVG,
        std::vector<size_t> {5, 5},  // kernel
        std::vector<size_t> {2, 2},  // strides
        std::vector<size_t> {2, 2},  // padBegins
        std::vector<size_t> {3, 3},  // padEnds
        ngraph::op::RoundingType::CEIL,
        ngraph::op::PadType::EXPLICIT,
        false  // placeholder value - exclude pad not applicable for max pooling
    ),
};

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_ExplicitPad_CeilRounding, KmbPoolingLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(avgPoolExplicitPadCeilRoundingParams),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Polling Cases ========== */

/*    ========== Valid Pad Rounding Not Applicable ========== */

const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3}, {5, 5}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1}, {2, 2}};

const auto allPoolsValidPadParams = ::testing::Combine(
    ::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::Values(InferenceEngine::SizeVector({0, 0})),
    ::testing::Values(InferenceEngine::SizeVector({0, 0})),
    ::testing::Values(ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
    ::testing::Values(ngraph::op::PadType::VALID),
    ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(smoke_MAX_and_AVGPool_ValidPad, KmbPoolingLayerTest,
    ::testing::Combine(
        allPoolsValidPadParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    PoolingLayerTest::getTestCaseName);

}  // namespace
