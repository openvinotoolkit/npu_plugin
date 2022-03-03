// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/functions.h"
#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

typedef std::tuple<InferenceEngine::Precision, InferenceEngine::Precision, InferenceEngine::SizeVector>
        KmbSwishTestParams;
class KmbSwishSingleInputTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<KmbSwishTestParams> {
    void SetUp() override {
        auto prms = GetParam();

        InferenceEngine::SizeVector inputShape;

        std::tie(inPrc, outPrc.front(), inputShape) = prms;

        inLayout = InferenceEngine::Layout::NCHW;

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto swish = std::make_shared<ngraph::op::v4::Swish>(paramOuts[0]);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(swish)};
        function = std::make_shared<ngraph::Function>(swish, params, "KmbSwishSingleInputTest");

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
    static std::string getTestCaseName(::testing::TestParamInfo<KmbSwishTestParams> obj) {
        auto params = obj.param;

        InferenceEngine::Precision ip, op;
        InferenceEngine::SizeVector inputShape;

        std::tie(ip, op, inputShape) = params;

        const std::string sep = "_";
        std::ostringstream result;

        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "InShape=" << VectorToString(inputShape) << sep;

        return result.str();
    }
};

TEST_P(KmbSwishSingleInputTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

const std::vector<InferenceEngine::Precision> in_prec = {InferenceEngine::Precision::UNSPECIFIED};

const std::vector<InferenceEngine::Precision> out_prec = {InferenceEngine::Precision::UNSPECIFIED};

const std::vector<InferenceEngine::SizeVector> inputShapes{{1, 3, 32, 32}, {1, 3, 200, 200}};

INSTANTIATE_TEST_CASE_P(smoke_SwishSingleInputTest, KmbSwishSingleInputTest,
                        ::testing::Combine(::testing::ValuesIn(in_prec), ::testing::ValuesIn(out_prec),
                                           ::testing::ValuesIn(inputShapes)),
                        KmbSwishSingleInputTest::getTestCaseName);

}  // namespace
