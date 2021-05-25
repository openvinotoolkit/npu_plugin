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

#include "ngraph_mcm_frontend/passes/convert_MVN6_to_MVN1.hpp"

#include <memory>
#include <ngraph/op/mvn.hpp>
#include <ngraph/op/constant.hpp>
#include <details/ie_exception.hpp>
#include "vpux/utils/core/error.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph/node.hpp"

ConvertMVN6toMVN1::ConvertMVN6toMVN1()
{
    auto mvn6 = ngraph::pattern::wrap_type<ngraph::op::v6::MVN>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m)
    {
        auto mvn6 = std::dynamic_pointer_cast<ngraph::op::v6::MVN>(m.get_match_root());
        if (!mvn6) {
            return false;
        }

        const auto input = mvn6->input_value(0);

        const bool normalize_variance = mvn6->get_normalize_variance();
        const float eps = mvn6->get_eps();
        const auto eps_mode = mvn6->get_eps_mode();

        if (normalize_variance && eps_mode != ngraph::op::MVNEpsMode::OUTSIDE_SQRT)
        {
            VPUX_THROW("MVN layer supports only OUTSIDE_SQRT eps_mode");
        }

        auto const_axes = std::dynamic_pointer_cast<ngraph::op::Constant>(mvn6->input(1).get_source_output().get_node_shared_ptr());
        IE_ASSERT(nullptr != const_axes);
        auto axes = const_axes->cast_vector<int32_t>();
        
        const auto dims_count = input.get_partial_shape().get_shape().size();
        VPUX_THROW_UNLESS(dims_count == 4, "MVN layer supports only 4D case");

        std::ostringstream ostr;
        for (auto &it: axes)
        {
            ostr << it << ", ";
            it = it < 0 ? it + dims_count : it; 
        }

        std::sort(axes.begin(), axes.end());

        bool across_channels = false;

        if (axes.size() == 3 && axes[0] == 1 && axes[1] == 2 && axes[2] == 3)
            across_channels = true;
        else if (axes.size() == 2 && axes[0] == 2 && axes[1] == 3)
            across_channels = false;
        else
            VPUX_THROW("MVN layer doesn't support axes '{0}', only normalization across channel or spatial dimension", ostr.str());
      
        const auto mcmMvn1 = std::make_shared<ngraph::op::v0::MVN>(input, across_channels, normalize_variance, (double)(eps));
        mcmMvn1->set_friendly_name(mvn6->get_friendly_name());

        ngraph::replace_node(mvn6, mcmMvn1);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mvn6, "ConvertMVN6toMVN1");
    register_matcher(m, callback);
}

// clang-format on
