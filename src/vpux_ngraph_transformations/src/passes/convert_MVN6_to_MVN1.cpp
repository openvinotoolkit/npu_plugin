//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "vpux/passes/convert_MVN6_to_MVN1.hpp"

#include <memory>
#include <ngraph/op/mvn.hpp>
#include <ngraph/op/constant.hpp>
#include <details/ie_exception.hpp>
#include "vpux/utils/core/error.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph/node.hpp"
#include "ngraph/log.hpp"

namespace vpux {

namespace passes {

ConvertMVN6toMVN1::ConvertMVN6toMVN1()
{
    auto mvn6 = ngraph::pattern::wrap_type<ngraph::op::v6::MVN>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m)
    {
        auto mvn6 = std::dynamic_pointer_cast<ngraph::op::v6::MVN>(m.get_match_root());
        if (!mvn6) {
            return false;
        }
        const auto eps_mode = mvn6->get_eps_mode();
        
        if(eps_mode != ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
            // MVN-1 does not support inside_sqrt eps mode, in this case we should do MVN6Decomposition pass
            // Disable temporarily to enable the BDK3 ModNet.
            NGRAPH_WARN << "MVN-1 does not support inside_sqrt eps mode.";
        }

        const auto input = mvn6->input_value(0);

        const bool normalize_variance = mvn6->get_normalize_variance();
        const float eps = mvn6->get_eps();

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
        else {
            //MVN-1 layer supports only normalization across channel or spatial dimension, in this case we should do MVN6Decomposition pass
            return false;
        }
      
        const auto Mvn1 = std::make_shared<ngraph::op::v0::MVN>(input, across_channels, normalize_variance, (double)(eps));
        Mvn1->set_friendly_name(mvn6->get_friendly_name());

        ngraph::replace_node(mvn6, Mvn1);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mvn6, "ConvertMVN6toMVN1");
    register_matcher(m, callback);
}

}  // namespace passes
}  // namespace vpux
// clang-format on
