// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/proposal.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbProposalLayerTest : public ProposalLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

/*    const normalize_type normalize = true;
    const feat_stride_type feat_stride = 1;
    const box_size_scale_type box_size_scale = 2.0f;
    const box_coordinate_scale_type box_coordinate_scale = 2.0f;

    void SetUp() override {
    proposalSpecificParams proposalParams;
    std::vector<float> img_info = {225.0f, 225.0f, 1.0f};

    std::tie(proposalParams, targetDevice) = this->GetParam();
    base_size_type base_size;
    pre_nms_topn_type pre_nms_topn;
    post_nms_topn_type post_nms_topn;
    nms_thresh_type nms_thresh;
    min_size_type min_size;
    ratio_type ratio;
    scale_type scale;
    clip_before_nms_type clip_before_nms;
    clip_after_nms_type clip_after_nms;
    framework_type framework;

    std::tie(base_size, pre_nms_topn,
             post_nms_topn,
             nms_thresh,
             min_size,
             ratio,
             scale,
             clip_before_nms,
             clip_after_nms,
             framework) = proposalParams;

    size_t bottom_w = base_size;
    size_t bottom_h = base_size;
    size_t num_anchors = ratio.size() * scale.size();

    std::vector<size_t> scoresShape = {1, 2 * num_anchors, bottom_h, bottom_w};
    std::vector<size_t> boxesShape  = {1, 4 * num_anchors, bottom_h, bottom_w};
    std::vector<size_t> imageInfoShape = {3};

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP16);
        // a_ and b_ are a workaround to solve alphabetic param sorting that destroys ordering
    auto params = ngraph::builder::makeParams(ngPrc, {{"a_scores", scoresShape}, {"b_boxes", boxesShape}});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto proposal = std::dynamic_pointer_cast<ngraph::opset4::Proposal>(
             ngraph::builder::makeProposal(paramOuts[0], paramOuts[1], img_info, ngPrc,
                                           base_size,
                                           pre_nms_topn,
                                           post_nms_topn,
                                           nms_thresh,
                                           feat_stride,
                                           min_size,
                                           ratio,
                                           scale,
                                           clip_before_nms,
                                           clip_after_nms,
                                           normalize,
                                           box_size_scale,
                                           box_coordinate_scale,
                                           framework));

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(proposal->output(0))};
    function = std::make_shared<ngraph::Function>(results, params, "proposal");
}*/

protected:
    void Validate() override {
        LayerTestsUtils::KmbLayerTestsCommon::Validate();
    }
};

TEST_P(KmbProposalLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbProposalLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

/* ============= Proposal ============= */
const std::vector<base_size_type> base_size_ = {16};
const std::vector<pre_nms_topn_type> pre_nms_topn_ = {100};
const std::vector<post_nms_topn_type> post_nms_topn_ = {100};
const std::vector<nms_thresh_type> nms_thresh_ = {0.7f};
const std::vector<min_size_type> min_size_ = {1};
const std::vector<ratio_type> ratio_ = {{1.0f, 2.0f}};
const std::vector<scale_type> scale_ = {{1.2f, 1.5f}};
const std::vector<clip_before_nms_type> clip_before_nms_ = {false};
const std::vector<clip_after_nms_type> clip_after_nms_ = {false};

// empty string corresponds to Caffe framework
const std::vector<framework_type> framework_ = {""};

const auto proposalParams = ::testing::Combine(
    ::testing::ValuesIn(base_size_),
    ::testing::ValuesIn(pre_nms_topn_),
    ::testing::ValuesIn(post_nms_topn_),
    ::testing::ValuesIn(nms_thresh_),
    ::testing::ValuesIn(min_size_),
    ::testing::ValuesIn(ratio_),
    ::testing::ValuesIn(scale_),
    ::testing::ValuesIn(clip_before_nms_),
    ::testing::ValuesIn(clip_after_nms_),
    ::testing::ValuesIn(framework_)
);

// Test fails with error:
// C++ exception with description "Size of dims(1) and format(NHWC) are inconsistent.
// openvino/inference-engine/src/inference_engine/ie_layouts.cpp:138" thrown in the test body.
// [Track number: S#40339]
INSTANTIATE_TEST_CASE_P(smoke_Proposal_tests, KmbProposalLayerTest,
                        ::testing::Combine(
                            proposalParams,
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbProposalLayerTest::getTestCaseName
);

}  // namespace
