//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/functions.h"
#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

typedef std::tuple<InferenceEngine::Precision, InferenceEngine::Precision, InferenceEngine::SizeVector,
                   InferenceEngine::SizeVector, ngraph::Strides, ngraph::Strides,
                   std::pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff>>
        CMajorConvNHWCTestParams;
class VPUXCMajorConvNHWCTest_VPU3700 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<CMajorConvNHWCTestParams> {
    //[Track number: E#26428]
    void SkipBeforeLoad() override {
        if (getBackendName(*getCore()) == "VPUAL") {
            throw LayerTestsUtils::KmbSkipTestException("LoadNetwork throws an exception");
        }
    }

    void ConfigureNetwork() override {
        auto params = GetParam();

        InferenceEngine::Precision ip, op;
        std::tie(ip, op, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = params;

        cnnNetwork.getInputsInfo().begin()->second->setPrecision(ip);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(op);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    void SetUp() override {
        auto prms = GetParam();

        InferenceEngine::SizeVector inputShape;
        InferenceEngine::SizeVector weightsShape;
        ngraph::Strides strides;
        std::pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff> pads;
        ngraph::Strides dilations;

        std::tie(std::ignore, std::ignore, inputShape, weightsShape, strides, dilations, pads) = prms;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto weightsU8 =
                ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

        const auto conv = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weightsFP32, strides, pads.first,
                                                                        pads.second, dilations);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(conv)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXCMajorConvNHWCTest");

        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        threshold = 0.1f;
    }

    template <typename T>
    static std::string VectorToString(std::vector<T> v) {
        std::ostringstream res;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) {
                res << ",";
            } else {
                res << "{";
            }

            res << v[i];
        }
        res << "}";
        return res.str();
    }

public:
    static std::string getTestCaseName(::testing::TestParamInfo<CMajorConvNHWCTestParams> obj) {
        auto params = obj.param;

        InferenceEngine::Precision ip, op;
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::SizeVector weightsShape;
        ngraph::Strides strides;
        std::pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff> pads;
        ngraph::Strides dilations;
        std::tie(ip, op, inputShape, weightsShape, strides, dilations, pads) = params;

        const std::string sep = "_";
        std::ostringstream result;

        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "InShape=" << VectorToString(inputShape) << sep;
        result << "WeightsShape=" << VectorToString(weightsShape) << sep;
        result << "Strides=" << VectorToString(strides) << sep;
        result << "Dilations=" << VectorToString(dilations) << sep;
        result << "Padding=" << VectorToString(pads.first) << "," << VectorToString(pads.second);

        return result.str();
    }
};

TEST_P(VPUXCMajorConvNHWCTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<InferenceEngine::Precision> prec = {InferenceEngine::Precision::FP16,
                                                      InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8};

const std::vector<InferenceEngine::SizeVector> inputShapes{{1, 3, 32, 32}};

const std::vector<InferenceEngine::SizeVector> weightShapes{{16, 3, 1, 1}};

const std::vector<ngraph::Strides> strides{{1, 1}};

const std::vector<ngraph::Strides> dilations{{1, 1}};

const std::vector<std::pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff>> pads{{{0, 0}, {0, 0}}};

/* NOTE: Tests have not yet run on actual device, because of CI instability. Only LoadNetwork phase was done. */
INSTANTIATE_TEST_CASE_P(smoke_Permute_NHWC_To_NCHW, VPUXCMajorConvNHWCTest_VPU3700,
                        ::testing::Combine(::testing::ValuesIn(prec), ::testing::ValuesIn(prec),
                                           ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(weightShapes),
                                           ::testing::ValuesIn(strides), ::testing::ValuesIn(dilations),
                                           ::testing::ValuesIn(pads)),
                        VPUXCMajorConvNHWCTest_VPU3700::getTestCaseName);

}  // namespace
