//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/propagate_fq.hpp"
#include <memory>
#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/rt_info.hpp>
#include "vpux/quantization_helpers.hpp"

namespace vpux {
namespace passes {

static void propogate_fq(std::shared_ptr<ngraph::Node> fq_node, int& copy_num);

std::shared_ptr<ngraph::Node> clone_up_fq_node(std::shared_ptr<ngraph::Node> fq_node,
                                               const ngraph::Output<ngraph::Node>& output, int& copy_num) {
    copy_num++;
    auto const_in_low = fq_node->input_value(1).get_node_shared_ptr()->clone_with_new_inputs({});
    auto const_in_high = fq_node->input_value(2).get_node_shared_ptr()->clone_with_new_inputs({});
    const_in_low->set_friendly_name(fq_node->input_value(1).get_node_shared_ptr()->get_friendly_name() + "_copy_" +
                                    std::to_string(copy_num));
    const_in_high->set_friendly_name(fq_node->input_value(2).get_node_shared_ptr()->get_friendly_name() + "_copy_" +
                                     std::to_string(copy_num));
    auto clone_fq_node =
            fq_node->clone_with_new_inputs({output, const_in_low, const_in_high, const_in_low, const_in_high});
    clone_fq_node->set_friendly_name(fq_node->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    return clone_fq_node;
}

std::shared_ptr<ngraph::Node> clone_down_fq_node(std::shared_ptr<ngraph::Node> fq_node,
                                                 const ngraph::Output<ngraph::Node>& output, int& copy_num) {
    copy_num++;
    auto const_out_low = fq_node->input_value(3).get_node_shared_ptr()->clone_with_new_inputs({});
    auto const_out_high = fq_node->input_value(4).get_node_shared_ptr()->clone_with_new_inputs({});
    const_out_low->set_friendly_name(fq_node->input_value(3).get_node_shared_ptr()->get_friendly_name() + "_copy_" +
                                     std::to_string(copy_num));
    const_out_high->set_friendly_name(fq_node->input_value(4).get_node_shared_ptr()->get_friendly_name() + "_copy_" +
                                      std::to_string(copy_num));
    auto clone_fq_node =
            fq_node->clone_with_new_inputs({output, const_out_low, const_out_high, const_out_low, const_out_high});
    clone_fq_node->set_friendly_name(fq_node->get_friendly_name() + "_copy_" + std::to_string(copy_num));
    return clone_fq_node;
}

static bool all_consumers_has_fq(const ngraph::Output<ngraph::Node>& node_output) {
    for (auto output : node_output.get_target_inputs()) {
        const auto output_as_fq =
                std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(output.get_node()->shared_from_this());
        if (output_as_fq == nullptr) {
            return false;
        }
    }
    return true;
}

static void propagate_down(std::shared_ptr<ngraph::Node> fq_node, std::shared_ptr<ngraph::Node> node, int& copy_num) {
    if (is_fq_agnostic(node)) {
        for (const auto& node_output : node->outputs()) {
            if (all_consumers_has_fq(node_output))
                continue;
            auto consumers = node_output.get_target_inputs();
            if (std::all_of(consumers.begin(), consumers.end(), [](const auto consumer) {
                    return std::dynamic_pointer_cast<ngraph::op::v0::Result>(consumer.get_node()->shared_from_this()) !=
                           nullptr;
                })) {
                continue;
            }

            auto new_fq_node = clone_down_fq_node(fq_node, node_output, copy_num);
            for (auto consumer : consumers) {
                // if FQ already exist - we don't need to generate a copy
                // also we can't insert FQ before Result node to keep output layer search logic
                const auto consumer_as_fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(
                        consumer.get_node()->shared_from_this());
                const auto consumer_as_result =
                        std::dynamic_pointer_cast<ngraph::op::v0::Result>(consumer.get_node()->shared_from_this());
                if (consumer_as_fq == nullptr && consumer_as_result == nullptr) {
                    propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
                    consumer.replace_source_output(new_fq_node);
                }
            }

            propogate_fq(new_fq_node, copy_num);
        }
    }
}

static void propagate_up(std::shared_ptr<ngraph::Node> fq_node, std::shared_ptr<ngraph::Node> node, int& copy_num) {
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

            auto new_fq_node = clone_up_fq_node(fq_node, input, copy_num);
            propagate_up(fq_node, input.get_node_shared_ptr(), copy_num);
            for (auto consumer : input.get_target_inputs()) {
                // if FQ already exist - we don't need to generate a copy
                // also we can't insert FQ before Result node to keep output layer search logic
                const auto consumer_as_fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(
                        consumer.get_node()->shared_from_this());
                const auto consumer_as_result =
                        std::dynamic_pointer_cast<ngraph::op::v0::Result>(consumer.get_node()->shared_from_this());
                if (consumer_as_fq == nullptr && consumer_as_result == nullptr) {
                    propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
                    consumer.replace_source_output(new_fq_node);
                }
            }
            propogate_fq(new_fq_node, copy_num);
        }
    }
}

static void propogate_fq(std::shared_ptr<ngraph::Node> fq_node, int& copy_num) {
    for (const auto& input : fq_node->input_values()) {
        propagate_up(fq_node, input.get_node_shared_ptr(), copy_num);
    }
    for (const auto& node_output : fq_node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
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

}  // namespace passes
}  // namespace vpux
