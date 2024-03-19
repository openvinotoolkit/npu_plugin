//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "shared_tests_instances/vpu_ov1_layer_test.hpp"
#include "single_layer_tests/non_max_suppression.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

class NmsLayerTestCommon : public NmsLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
public:
    void GenerateInputs() override {
        size_t it = 0;
        for (const auto& input : cnnNetwork.getInputsInfo()) {
            const auto& info = input.second;
            Blob::Ptr blob;
            if ((it == 0) || (it == 1)) {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                uint32_t range = 1;
                uint32_t resolution = 1000;
                if (it == 0) {  // default GenerateInput parameters
                    range = 10;
                    resolution = 1;
                }
                if (info->getTensorDesc().getPrecision() == Precision::FP32) {
                    fillDataRandomFloatWithFp16Precision(blob, range, 0, resolution);
                } else {
                    ov::test::utils::fill_data_random_float<InferenceEngine::Precision::FP16>(blob, range, 0,
                                                                                              resolution);
                }
                if (it == 0) {
                    sortCorner(blob);
                }
            } else {
                blob = GenerateInput(*info);
            }
            inputs.push_back(blob);
            it++;
        }
    }

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override {
        //  convert back to int32 if was changed to fp32. Change requested by change inside vpu_ov1_layer_test.cpp "//
        //  TODO: Remove after C#-101214 convertDataToFP32(actualOutputs);"
        auto actualOutputsConvBack = actualOutputs;
        actualOutputsConvBack[0] =
                FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::I32>(actualOutputsConvBack[0]);
        NmsLayerTest::Compare(expectedOutputs, actualOutputsConvBack);
    }

private:
    void fillDataRandomFloatWithFp16Precision(InferenceEngine::Blob::Ptr& blob, const uint32_t range,
                                              int32_t start_from, const int32_t k) {
        std::default_random_engine random(1);
        std::uniform_int_distribution<int32_t> distribution(k * start_from, k * (start_from + range));
        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        for (size_t i = 0; i < blob->size(); i++) {
            auto value = static_cast<float>(distribution(random));
            value /= static_cast<float>(k);
            ngraph::float16 fp16Val = ngraph::float16(value);
            rawBlobDataPtr[i] = static_cast<float>(fp16Val);
        }
    }
    void sortCorner(InferenceEngine::Blob::Ptr& blob) {
        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        for (size_t i = 0; i < blob->size(); i += 4) {
            float y1 = rawBlobDataPtr[i + 0];
            float x1 = rawBlobDataPtr[i + 1];
            float y2 = rawBlobDataPtr[i + 2];
            float x2 = rawBlobDataPtr[i + 3];

            float ymin = std::min(y1, y2);
            float ymax = std::max(y1, y2);
            float xmin = std::min(x1, x2);
            float xmax = std::max(x1, x2);

            rawBlobDataPtr[i + 0] = ymin;
            rawBlobDataPtr[i + 1] = xmin;
            rawBlobDataPtr[i + 2] = ymax;
            rawBlobDataPtr[i + 3] = xmax;
        }
    }

protected:
    void SetUp() override {
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
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(boxesShape)),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(scoresShape))};

        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));

        auto nms = builder::makeNms(paramOuts[0], paramOuts[1], convertIE2nGraphPrc(maxBoxPrec),
                                    convertIE2nGraphPrc(thrPrec), maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma,
                                    boxEncoding == ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                    sortResDescend, outType, ngraph::builder::NmsVersion::NmsVersion9);
        function = std::make_shared<Function>(nms, params, "NMS");
    }
};

class NmsLayerTest_NPU3700 : public NmsLayerTestCommon {};
class NmsLayerTest_NPU3720 : public NmsLayerTestCommon {};

TEST_P(NmsLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(NmsLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

namespace {

using namespace ngraph;
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

const std::vector<InputShapeParams> inShapeParams = {
        InputShapeParams{1, 80, 1},   // standard params usage 90% of conformance tests
        InputShapeParams{1, 40, 20},  // 1 usage style
        InputShapeParams{3, 30, 18},  // for check remain posibility
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 15};
const std::vector<float> iouThreshold = {0.3f, 0.7f};
const std::vector<float> scoreThreshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
        op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
        op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
};
const std::vector<bool> sortResDesc = {false, true};
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

// ------- NPU3700/3720 full scope -------
const auto nmsParams = ::testing::Combine(
        ::testing::ValuesIn(inShapeParams),
        ::testing::Combine(::testing::ValuesIn(paramsPrec), ::testing::ValuesIn(maxBoxPrec),
                           ::testing::ValuesIn(thrPrec)),
        ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold),
        ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
        ::testing::ValuesIn(outType), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_NmsLayerTest, NmsLayerTest_NPU3700, nmsParams, NmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_NmsLayerTest, NmsLayerTest_NPU3720, nmsParams,
                        NmsLayerTest::getTestCaseName);

// --------- NPU3720 precommit scope ---------
const std::vector<InputShapeParams> inShapeParamsSmoke = {InputShapeParams{2, 9, 12}};
const std::vector<int32_t> maxOutBoxPerClassSmoke = {5};
const std::vector<float> iouThresholdSmoke = {0.3f};
const std::vector<float> scoreThresholdSmoke = {0.3f};
const std::vector<float> sigmaThresholdSmoke = {0.0f, 0.5f};
const std::vector<op::v5::NonMaxSuppression::BoxEncodingType> encodTypeSmoke = {
        op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const auto nmsParamsSmoke =
        testing::Combine(testing::ValuesIn(inShapeParamsSmoke),
                         ::testing::Combine(::testing::ValuesIn(paramsPrec), ::testing::ValuesIn(maxBoxPrec),
                                            ::testing::ValuesIn(thrPrec)),
                         ::testing::ValuesIn(maxOutBoxPerClassSmoke), ::testing::ValuesIn(iouThresholdSmoke),
                         ::testing::ValuesIn(scoreThresholdSmoke), ::testing::ValuesIn(sigmaThresholdSmoke),
                         ::testing::ValuesIn(encodTypeSmoke), ::testing::ValuesIn(sortResDesc),
                         ::testing::ValuesIn(outType), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_NmsLayerTest, NmsLayerTest_NPU3720, nmsParamsSmoke,
                        NmsLayerTest::getTestCaseName);
}  // namespace
