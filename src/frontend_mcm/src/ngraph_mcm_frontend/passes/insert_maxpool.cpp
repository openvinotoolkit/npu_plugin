//
// Copyright 2021 Intel Corporation.
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

// clang-format on
#include "ngraph_mcm_frontend/passes/insert_maxpool.hpp"
#include <details/ie_exception.hpp>

#include <memory>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/variadic_split.hpp>
#include <ngraph/type/element_type.hpp>

#include "ngraph_mcm_frontend/quantization_helpers.hpp"

namespace {
bool hasVarSplitParent(const std::shared_ptr<ngraph::Node>& fq_node) {
    const auto input_values = fq_node->input_values();
    const auto first_input = input_values.at(0).get_node_shared_ptr();
    return std::dynamic_pointer_cast<ngraph::op::v1::VariadicSplit>(first_input) != nullptr;
}

void insertPooling(const std::shared_ptr<ngraph::Node>& fq_node) {
    const auto input_values = fq_node->input_values();
    const auto strides = ngraph::Strides{1, 1};
    const auto pads_begin = ngraph::Shape{0, 0};
    const auto pads_end = ngraph::Shape{0, 0};
    const auto kernel = ngraph::Shape{1, 1};
    const auto rounding_mode = ngraph::op::RoundingType::FLOOR;
    const auto auto_pad = ngraph::op::PadType::EXPLICIT;
    const auto max_pool =
            std::make_shared<ngraph::op::v1::MaxPool>(input_values.at(0).get_node_shared_ptr()->output(0), strides,
                                                      pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    const auto input_low = input_values.at(1).get_node_shared_ptr();
    const auto input_high = input_values.at(2).get_node_shared_ptr();
    const auto output_low = input_values.at(3).get_node_shared_ptr();
    const auto output_high = input_values.at(4).get_node_shared_ptr();

    const auto old_fq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(fq_node);
    // this check must never fail since all non-FQ nodes have already been filtered out at this point
    if (old_fq == nullptr) {
        THROW_IE_EXCEPTION << "ForceFP16Split::insertPooling: failed to cast fake quantize pointer";
    }
    auto levels = old_fq->get_levels();
    auto new_fq = std::make_shared<ngraph::op::FakeQuantize>(max_pool->output(0), input_low, input_high, output_low,
                                                             output_high, levels);
    ngraph::replace_node(fq_node, new_fq);
}
}  // namespace

bool InsertMaxPool::run_on_node(std::shared_ptr<ngraph::Node> node) {
    // replace { VarSplit -> FQ -> Concat } with { VarSplit -> MaxPool -> FQ -> Concat }
    // this is required to cast VarSplit output to float16
    // FIXME: cast VarSplit output to float16 in some less cryptic way
    if (std::dynamic_pointer_cast<ngraph::op::v0::Concat>(node) != nullptr) {
        // get all FQ inputs of Concat
        auto fq_concat = getInputsFQ(node);
        decltype(fq_concat) split_fq_concat;
        // filter FQs with VarSplit input
        std::copy_if(fq_concat.begin(), fq_concat.end(), std::back_inserter(split_fq_concat), hasVarSplitParent);
        // insert max pool between VarSplit and FQ
        std::for_each(split_fq_concat.begin(), split_fq_concat.end(), insertPooling);

        return true;
    }
    return false;
}
