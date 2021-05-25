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

#include "ngraph_mcm_frontend/passes/collapse_concat_chain.hpp"

#include <memory>

#include "ngraph/op/concat.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/rt_info.hpp"

ngraph::pass::CollapseConcats0238::CollapseConcats0238() {
    // Pass for replace subgraph:
    //             FQ      FQ        FQ      FQ            //
    //    FQ         Concat            Concat       FQ     //
    //                         Concat                      //
    //
    // with subgraph:
    //                                                     //
    //    FQ       FQ      FQ       FQ      FQ      FQ     //
    //                         Concat                      //
    //
    // build concat1
    auto concat1_fq1_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq1_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq1_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq1_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq1_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq1 = std::make_shared<ngraph::opset1::FakeQuantize>(concat1_fq1_1, concat1_fq1_2, concat1_fq1_3,
                                                                      concat1_fq1_4, concat1_fq1_5, 1);

    auto concat1_fq2_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq2_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq2_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq2_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq2_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat1_fq2 = std::make_shared<ngraph::opset1::FakeQuantize>(concat1_fq2_1, concat1_fq2_2, concat1_fq2_3,
                                                                      concat1_fq2_4, concat1_fq2_5, 1);

    auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({concat1_fq1, concat1_fq2}), 1);

    // build concat2
    auto concat2_fq1_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq1_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq1_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq1_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq1_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq1 = std::make_shared<ngraph::opset1::FakeQuantize>(concat2_fq1_1, concat2_fq1_2, concat2_fq1_3,
                                                                      concat2_fq1_4, concat2_fq1_5, 1);

    auto concat2_fq2_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq2_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq2_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq2_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq2_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto concat2_fq2 = std::make_shared<ngraph::opset1::FakeQuantize>(concat2_fq2_1, concat2_fq2_2, concat2_fq2_3,
                                                                      concat2_fq2_4, concat2_fq2_5, 1);

    auto concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector({concat2_fq1, concat2_fq2}), 1);

    // build left fq
    auto datafq1_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq1_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq1_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq1_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq1_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto fq1 = std::make_shared<ngraph::opset1::FakeQuantize>(datafq1_1, datafq1_2, datafq1_3, datafq1_4, datafq1_5, 1);

    // build right fq
    auto datafq2_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq2_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq2_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq2_4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto datafq2_5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto fq2 = std::make_shared<ngraph::opset1::FakeQuantize>(datafq2_1, datafq2_2, datafq2_3, datafq2_4, datafq2_5, 1);

    // build main concat
    ngraph::OutputVector concatsNode{fq1, concat1, concat2, fq2};
    auto concat = std::make_shared<ngraph::opset1::Concat>(concatsNode, 1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(m.get_match_root());
        if (!concat)
            return false;

        if (concat->inputs().size() != 4)
            return false;

        auto concat1 = std::dynamic_pointer_cast<ngraph::opset1::Concat>(
                concat->input(1).get_source_output().get_node_shared_ptr());
        if (!concat1 || concat1->inputs().size() != 2)
            return false;
        auto concat2 = std::dynamic_pointer_cast<ngraph::opset1::Concat>(
                concat->input(2).get_source_output().get_node_shared_ptr());
        if (!concat2 || concat2->inputs().size() != 2)
            return false;

        if (concat->get_axis() != concat1->get_axis() || concat->get_axis() != concat2->get_axis())
            return false;

        auto newConcat = std::make_shared<ngraph::opset1::Concat>(
                ngraph::OutputVector{concat->input(0).get_source_output(), concat1->input(0).get_source_output(),
                                     concat1->input(1).get_source_output(), concat2->input(0).get_source_output(),
                                     concat2->input(1).get_source_output(), concat->input(3).get_source_output()},
                1);

        newConcat->set_friendly_name(concat->get_friendly_name() + "_merged");
        ngraph::copy_runtime_info({concat, concat1, concat2}, newConcat);
        ngraph::replace_node(concat, newConcat);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "mcmAdaptation.CollapseConcats0238");
    this->register_matcher(m, callback);
}
