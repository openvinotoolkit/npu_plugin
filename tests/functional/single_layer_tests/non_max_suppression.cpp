//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vector>

#include "shared_tests_instances/kmb_layer_test.hpp"
#include "single_layer_tests/non_max_suppression.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

class KmbNmsLayerTest : public NmsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
private:
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override;

protected:
    void SetUp() override;
};

class KmbNmsLayerTest_VPU3720 : public KmbNmsLayerTest {};

void KmbNmsLayerTest::SetUp() {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    size_t maxOutBoxesPerClass;
    float iouThr, scoreThr, softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    bool sortResDescend;
    element::Type outType;
    std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding,
             sortResDescend, outType, targetDevice) = this->GetParam();

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
    auto ngPrc = convertIE2nGraphPrc(paramsPrec);
    auto params = builder::makeParams(ngPrc, {boxesShape, scoresShape});
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));

    auto nms = builder::makeNms(paramOuts[0], paramOuts[1], convertIE2nGraphPrc(maxBoxPrec),
                                convertIE2nGraphPrc(thrPrec), maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma,
                                boxEncoding == ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER, sortResDescend,
                                outType);
    function = std::make_shared<Function>(nms, params, "NMS");
}

void KmbNmsLayerTest::Compare(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
        const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        const auto& expectedBuffer = expected.second.data();
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const uint8_t*>();

        auto k = static_cast<float>(expected.first.size()) / actual->getTensorDesc().getPrecision().size();
        // W/A for int4, uint4
        if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
            k /= 2;
        }
        if (outputIndex == 2) {
            if (expected.second.size() != k * actual->byteSize())
                throw std::runtime_error("Expected and actual size 3rd output have different size");
        }

        const auto& precision = actual->getTensorDesc().getPrecision();
        size_t size = expected.second.size() / (k * actual->getTensorDesc().getPrecision().size());
        switch (precision) {
        case InferenceEngine::Precision::FP32: {
            switch (expected.first) {
            case ngraph::element::Type_t::f32:
                LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const float*>(expectedBuffer),
                                                           reinterpret_cast<const float*>(actualBuffer), size,
                                                           threshold);
                break;
            case ngraph::element::Type_t::f64:
                LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const double*>(expectedBuffer),
                                                           reinterpret_cast<const float*>(actualBuffer), size,
                                                           threshold);
                break;
            default:
                break;
            }

            const auto fBuffer = lockedMemory.as<const float*>();
            for (int i = size; i < actual->size(); i++) {
                ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
            }
            break;
        }
        case InferenceEngine::Precision::I32: {
            switch (expected.first) {
            case ngraph::element::Type_t::i32:
                LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int32_t*>(expectedBuffer),
                                                           reinterpret_cast<const int32_t*>(actualBuffer), size, 0);
                break;
            case ngraph::element::Type_t::i64:
                LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int64_t*>(expectedBuffer),
                                                           reinterpret_cast<const int32_t*>(actualBuffer), size, 0);
                break;
            default:
                break;
            }

            const auto iBuffer = lockedMemory.as<const int*>();
            for (int i = size; i < actual->size(); i++) {
                ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
            }
            break;
        }
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

TEST_P(KmbNmsLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(KmbNmsLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

namespace {

using namespace ngraph;
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

const std::vector<InputShapeParams> inShapeParams = {
        InputShapeParams{3, 100, 5},
        InputShapeParams{1, 10, 50},
        InputShapeParams{2, 50, 50},
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> iouThreshold = {0.3f, 0.7f};
const std::vector<float> scoreThreshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f};
const std::vector<op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
        op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
        op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
};
const std::vector<bool> sortResDesc = {false};
const std::vector<element::Type> outType = {element::i32};
std::vector<InferenceEngine::Precision> paramsPrec = {
        InferenceEngine::Precision::FP32,
};
std::vector<InferenceEngine::Precision> maxBoxPrec = {
        InferenceEngine::Precision::I32,
};
std::vector<InferenceEngine::Precision> thrPrec = {
        InferenceEngine::Precision::FP16,
};

const auto nmsParams = ::testing::Combine(
        ::testing::ValuesIn(inShapeParams),
        ::testing::Combine(::testing::ValuesIn(paramsPrec), ::testing::ValuesIn(maxBoxPrec),
                           ::testing::ValuesIn(thrPrec)),
        ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold),
        ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
        ::testing::ValuesIn(outType), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_NmsLayerTest, KmbNmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_NmsLayerTest, KmbNmsLayerTest_VPU3720, nmsParams,
                        NmsLayerTest::getTestCaseName);

}  // namespace
