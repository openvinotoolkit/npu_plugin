//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <memory>

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

TEST(MLIR_IE_FrontEndTest, ProposalLayer) {
    ngraph::op::ProposalAttrs attr;
    attr.base_size = 256;
    attr.pre_nms_topn = 2147483647;
    attr.post_nms_topn = 100;
    attr.nms_thresh = 0.699999988079f;
    attr.feat_stride = 16;
    attr.min_size = 1;
    attr.ratio = {0.5f, 1.0f, 2.0f};
    attr.scale = {0.25f, 0.5f, 1.0f, 2.0f};
    attr.clip_before_nms = true;
    attr.clip_after_nms = false;
    attr.normalize = true;
    attr.box_size_scale = 5.0f;
    attr.box_coordinate_scale = 10.0f;
    attr.framework = "tensorflow";
    attr.infer_probs = false;

    std::shared_ptr<ngraph::Function> f;
    {
        auto prediction =
                std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 24, 38, 38});
        auto bboxes = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 48, 38, 38});
        std::vector<float> imageDetailsVector{480, 640, 1};
        auto imageDetails =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{3}, imageDetailsVector);
        auto proposal = std::make_shared<ngraph::opset1::Proposal>(prediction, bboxes, imageDetails, attr);
        proposal->set_friendly_name("Proposal");
        auto result = std::make_shared<ngraph::op::Result>(proposal);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{prediction, bboxes});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
