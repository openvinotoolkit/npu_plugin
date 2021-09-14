// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/subgraph/scaleshift.hpp>

#include <vector>

#include "kmb_layer_test.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

class KmbTransposeReshapeTransposeTest : public LayerTestsUtils::KmbLayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        std::vector<std::vector<size_t >> inputs {{1, 10, 2, 1}};
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP16);

        auto input = ngraph::builder::makeParams(ngPrc, {{1, 10, 2, 1}});

        auto permute1_params = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                      ngraph::Shape{4},
                                                                      std::vector<size_t>{0, 2, 3, 1});
        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(input[0], permute1_params);

        auto reshape1_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{4},
                                                                       std::vector<size_t>{1, 4, 1, 5});
        auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(permute1, reshape1_pattern, false);

        auto permute2_params = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                      ngraph::Shape{4},
                                                                      std::vector<size_t>{0, 3, 1, 2});
        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(reshape1, permute2_params);

        function = std::make_shared<ngraph::Function>(permute2, input, "permute_reshape_permute");
    }
};


class KmbTransposeReshapeTransposeTestMCM: public KmbTransposeReshapeTransposeTest {};
class KmbTransposeReshapeTransposeTestMLIR: public KmbTransposeReshapeTransposeTest {};

TEST_F(KmbTransposeReshapeTransposeTestMCM, CompareWithRefs) {
    Run();
}

TEST_F(KmbTransposeReshapeTransposeTestMLIR, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

}  // namespace SubgraphTestsDefinitions
