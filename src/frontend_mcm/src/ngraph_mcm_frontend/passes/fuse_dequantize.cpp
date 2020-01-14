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

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/passes/fuse_dequantize.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_dequantize.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/constant.hpp>
#include <memory>

namespace {

static bool processConv(std::shared_ptr<McmConv> conv) {
    bool modified = false;

    for (size_t i = 0; i < conv->get_input_size(); ++i) {
        if (const auto dequantize = std::dynamic_pointer_cast<McmDequantize>(conv->input_value(i).get_node_shared_ptr())) {
            const auto input = dequantize->input_value(0).get_node_shared_ptr();
            IE_ASSERT(input != nullptr);

            const auto scales = std::dynamic_pointer_cast<ngraph::op::Constant>(dequantize->input_value(1).get_node_shared_ptr());
            const auto zeroPoints = std::dynamic_pointer_cast<ngraph::op::Constant>(dequantize->input_value(2).get_node_shared_ptr());
            IE_ASSERT(scales != nullptr && zeroPoints != nullptr);

            const auto scalesData = scales->cast_vector<double>();
            const auto zeroPointsData = zeroPoints->cast_vector<int64_t>();

            const auto newQuantParams = makeQuantParams(zeroPointsData, scalesData);
            McmOpAttrs::setQuantParams(newQuantParams, input);

            conv->input(i).replace_source_output(input);

            modified = true;
        }
    }

    return modified;
}

}  // namespace

bool FuseDequantize::run_on_node(std::shared_ptr<ngraph::Node> node) {
    if (const auto conv = std::dynamic_pointer_cast<McmConv>(node)) {
        return processConv(conv);
    }

    return false;
}

#endif
// clang-format on
