//
// Copyright 2021 Intel Corporation.
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

// clang-format off

#include "vpux/passes/convert_Proposal4_to_Proposal1.hpp"

#include <memory>
#include <ngraph/op/proposal.hpp>
#include <ngraph/op/constant.hpp>
#include <details/ie_exception.hpp>
#include "vpux/utils/core/error.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph/node.hpp"
#include "ngraph/log.hpp"

namespace vpux {

namespace passes {

ConvertProposal4toProposal1::ConvertProposal4toProposal1()
{
    auto proposal4 = ngraph::pattern::wrap_type<ngraph::op::v4::Proposal>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m)
    {
        auto proposal4 = std::dynamic_pointer_cast<ngraph::op::v4::Proposal>(m.get_match_root());
        if (!proposal4) {
            return false;
        }
        const auto class_probs = proposal4->input_value(0);
        const auto bbox_deltas = proposal4->input_value(1);
        const auto image_shape = proposal4->input_value(2);
       // const auto ProposalAttrs attrs = get_attrs();
        const auto Proposal1 = std::make_shared<ngraph::op::v0::Proposal>(/*class_probs, bbox_deltas, image_shape, attrs*/);
        Proposal1->set_friendly_name(proposal4->get_friendly_name());
        //ngraph::copy_runtime_info(proposal4, Proposal1);
        ngraph::replace_node(proposal4, Proposal1);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal4, "ConvertProposal4toProposal1");
    register_matcher(m, callback);
}

}  // namespace passes
}  // namespace vpux
// clang-format on
