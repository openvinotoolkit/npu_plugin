//
// Copyright Intel Corporation.
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

// clang-format on

#include "ngraph_mcm_frontend/passes/remove_splitConcat.hpp"

#include <memory>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/split.hpp>
#include <ngraph/op/transpose.hpp>
#include <ngraph/op/variadic_split.hpp>
#include "ngraph/op/concat.hpp"

// Modnet Workaround: remove [split -> concat] subgraph if it is directly after input node
bool RemoveSplitConcat::run_on_node(std::shared_ptr<ngraph::Node> node) {
    const auto split_node = std::dynamic_pointer_cast<ngraph::op::v1::Split>(node);
    if (split_node != nullptr) {
        const auto parent_node = split_node->input_value(0).get_node_shared_ptr();
        const auto parameter_node = std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(parent_node);
        if (parameter_node == nullptr)
            return false;

        for (const auto& node_output : split_node->outputs()) {
            const auto target_inputs = node_output.get_target_inputs();
            if (!target_inputs.empty()) {
                const auto consumer_input = *target_inputs.begin();
                const auto consumer = consumer_input.get_node();
                if (consumer) {
                    const auto concat_node =
                            std::dynamic_pointer_cast<ngraph::op::v0::Concat>(consumer->shared_from_this());
                    if (concat_node != nullptr) {
                        replace_output_update_name(concat_node->output(0), split_node->input_value(0));
                        return true;
                    }
                }
            }
        }
    }

    return false;
}
