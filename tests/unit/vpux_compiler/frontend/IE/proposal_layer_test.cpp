//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
