// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/proposal.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbProposalLayerTest : public ProposalLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
protected:
    void Validate() override {
        LayerTestsUtils::KmbLayerTestsCommon::Validate();
    }
};

TEST_P(KmbProposalLayerTest, CompareWithRefs) {
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
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Proposal_tests, KmbProposalLayerTest,
                        ::testing::Combine(
                            proposalParams,
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        KmbProposalLayerTest::getTestCaseName
);

}  // namespace
