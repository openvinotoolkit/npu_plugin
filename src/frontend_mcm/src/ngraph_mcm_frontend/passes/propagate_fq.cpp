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

// clang-format off

#include "ngraph_mcm_frontend/passes/propagate_fq.hpp"
#include "ngraph_mcm_frontend/quantization_helpers.hpp"
#include <ngraph/op/fake_quantize.hpp>
#include <memory>
#include <ngraph/rt_info.hpp>
#include <ngraph/ops.hpp>

std::shared_ptr<ngraph::Node> clone_fq_node(std::shared_ptr<ngraph::Node> fq_node, const ngraph::Output<ngraph::Node> &input, int &copy_num) {
    copy_num++;
    auto const1 = fq_node->input_value(1).get_node_shared_ptr()->clone_with_new_inputs({});
    auto const2 = fq_node->input_value(2).get_node_shared_ptr()->clone_with_new_inputs({});
    auto const3 = fq_node->input_value(3).get_node_shared_ptr()->clone_with_new_inputs({});
    auto const4 = fq_node->input_value(4).get_node_shared_ptr()->clone_with_new_inputs({});
    const1->set_friendly_name(fq_node->input_value(1).get_node_shared_ptr()->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    const2->set_friendly_name(fq_node->input_value(2).get_node_shared_ptr()->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    const3->set_friendly_name(fq_node->input_value(3).get_node_shared_ptr()->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    const4->set_friendly_name(fq_node->input_value(4).get_node_shared_ptr()->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    auto clone_fq_node = fq_node->clone_with_new_inputs({input, const1, const2, const3, const4});
    clone_fq_node->set_friendly_name(fq_node->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    return clone_fq_node;
}

static bool all_consumers_has_fq(const ngraph::Output<ngraph::Node> &node_output) {
    for (auto output : node_output.get_target_inputs()) {
        const auto output_as_fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(output.get_node()->shared_from_this());
        if (output_as_fq == nullptr) {
            return false;
        }
    }
    return true;
}

static void propagate_down(std::shared_ptr<ngraph::Node> fq_node, std::shared_ptr<ngraph::Node> node, int &copy_num) {
    if (is_fq_agnostic(node)) {
        for (const auto &node_output : node->outputs()) {
            if (all_consumers_has_fq(node_output))
                continue;

            auto new_fq_node = clone_fq_node(fq_node, node_output, copy_num);
            for (auto consumer : node_output.get_target_inputs()) {
                const auto consumer_as_fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(consumer.get_node()->shared_from_this());
                if (consumer_as_fq == nullptr) {
                    propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
                    consumer.replace_source_output(new_fq_node);
                }
            }
        }
    }
}

static void propagate_up(std::shared_ptr<ngraph::Node> fq_node, std::shared_ptr<ngraph::Node> node, int &copy_num) {
    if (is_fq_agnostic(node)) {
        for (auto input : node->input_values()) {
            if (std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(input.get_node_shared_ptr()) != nullptr)
                continue;
            if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input.get_node_shared_ptr()) != nullptr)
                continue;
            if ((std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(node) != nullptr ||
                 std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node) != nullptr))
                if (input == node->input_values().back())
                    continue;

            auto new_fq_node = clone_fq_node(fq_node, input, copy_num);
            propagate_up(fq_node, input.get_node_shared_ptr(), copy_num);
            for (auto consumer : input.get_target_inputs()) {
                const auto consumer_as_fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(consumer.get_node()->shared_from_this());
                if (consumer_as_fq == nullptr) {
                    propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
                    consumer.replace_source_output(new_fq_node);
                }
            }
        }
    }
}

bool PropagateFQ::run_on_node(std::shared_ptr<ngraph::Node> node) {
    const auto fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(node);
    if (fq_node == nullptr)
        return false;

    int fq_copies_num = 0;
    for (const auto& input : fq_node->input_values()) {
        propagate_up(fq_node, input.get_node_shared_ptr(), fq_copies_num);
    }
    for (const auto& node_output : fq_node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            propagate_down(fq_node, consumer.get_node()->shared_from_this(), fq_copies_num);
        }
    }

    return fq_copies_num != 0;
}
