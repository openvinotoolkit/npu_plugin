//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/propagate_fq.hpp"
#include <memory>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/ops.hpp>
#include "vpux/quantization_helpers.hpp"

namespace vpux {
namespace passes {

static void propogate_fq(std::shared_ptr<ov::Node> fq_node, int& copy_num);

std::shared_ptr<ov::Node> clone_up_fq_node(std::shared_ptr<ov::Node> fq_node, const ov::Output<ov::Node>& output,
                                           int& copy_num) {
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

std::shared_ptr<ov::Node> clone_down_fq_node(std::shared_ptr<ov::Node> fq_node, const ov::Output<ov::Node>& output,
                                             int& copy_num) {
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

static bool all_consumers_has_fq_or_result(const ov::Output<ov::Node>& node_output) {
    for (auto& output : node_output.get_target_inputs()) {
        const auto output_as_fq =
                std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(output.get_node()->shared_from_this());
        const auto output_as_result =
                std::dynamic_pointer_cast<ov::op::v0::Result>(output.get_node()->shared_from_this());
        if (output_as_fq == nullptr && output_as_result == nullptr) {
            return false;
        }
    }
    return true;
}

static void propagate_down(std::shared_ptr<ov::Node> fq_node, std::shared_ptr<ov::Node> node, int& copy_num) {
    if (is_fq_agnostic(node)) {
        for (const auto& node_output : node->outputs()) {
            if (all_consumers_has_fq_or_result(node_output))
                continue;
            auto consumers = node_output.get_target_inputs();
            if (std::all_of(consumers.begin(), consumers.end(), [](const auto consumer) {
                    return std::dynamic_pointer_cast<ov::op::v0::Result>(consumer.get_node()->shared_from_this()) !=
                           nullptr;
                })) {
                continue;
            }

            const auto& new_fq_node = clone_down_fq_node(fq_node, node_output, copy_num);
            for (auto& consumer : consumers) {
                // if FQ already exist - we don't need to generate a copy
                // also we can't insert FQ before Result node to keep output layer search logic
                const auto consumer_as_fq =
                        std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(consumer.get_node()->shared_from_this());
                const auto consumer_as_result =
                        std::dynamic_pointer_cast<ov::op::v0::Result>(consumer.get_node()->shared_from_this());
                if (consumer_as_fq == nullptr && consumer_as_result == nullptr) {
                    propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
                    consumer.replace_source_output(new_fq_node);
                }
            }

            propogate_fq(new_fq_node, copy_num);
        }
    }
}

static bool all_consumers_has_same_fq(const std::shared_ptr<ov::Node>& node, const std::shared_ptr<ov::Node>& fq_node) {
    const auto get_constant_op = [](const std::shared_ptr<ov::Node>& node, size_t idx) {
        OPENVINO_ASSERT(idx < node->get_input_size());
        auto constant_op =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(node->input_value(idx).get_node_shared_ptr());
        OPENVINO_ASSERT(constant_op != nullptr);
        return constant_op;
    };

    const auto is_same_values = [](const std::shared_ptr<ov::op::v0::Constant>& const_node1,
                                   const std::shared_ptr<ov::op::v0::Constant>& const_node2) {
        auto node1_data = const_node1->cast_vector<float>();
        auto node2_data = const_node2->cast_vector<float>();

        if (node1_data.size() != node2_data.size()) {
            return false;
        }

        for (size_t idx = 0; idx < node1_data.size(); idx++) {
            if (std::abs(node1_data[idx] - node2_data[idx]) > std::numeric_limits<float>::epsilon()) {
                return false;
            }
        }
        return true;
    };

    auto in_low_base = get_constant_op(fq_node, 1);
    auto in_high_base = get_constant_op(fq_node, 2);
    auto out_low_base = get_constant_op(fq_node, 3);
    auto out_high_base = get_constant_op(fq_node, 4);

    for (auto& output : node->outputs()) {
        auto consumers = output.get_target_inputs();
        for (auto& consumer : consumers) {
            const auto sibling_fq =
                    std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(consumer.get_node()->shared_from_this());
            if (sibling_fq == nullptr) {
                return false;
            }

            if (sibling_fq == fq_node) {
                continue;
            }

            auto is_same_in_low = is_same_values(in_low_base, get_constant_op(sibling_fq, 1));
            auto is_same_in_high = is_same_values(in_high_base, get_constant_op(sibling_fq, 2));
            auto is_same_out_low = is_same_values(out_low_base, get_constant_op(sibling_fq, 3));
            auto is_same_out_high = is_same_values(out_high_base, get_constant_op(sibling_fq, 4));

            if (!is_same_in_low || !is_same_in_high || !is_same_out_low || !is_same_out_high) {
                return false;
            }
        }
    }

    return true;
}

static void propagate_up(std::shared_ptr<ov::Node> fq_node, std::shared_ptr<ov::Node> node, int& copy_num) {
    if (is_fq_agnostic(node) && all_consumers_has_same_fq(node, fq_node)) {
        for (auto& input : node->input_values()) {
            if (std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(input.get_node_shared_ptr()) != nullptr)
                continue;
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(input.get_node_shared_ptr()) != nullptr)
                continue;
            if ((std::dynamic_pointer_cast<ov::op::v0::Interpolate>(node) != nullptr ||
                 std::dynamic_pointer_cast<ov::op::v1::Reshape>(node) != nullptr))
                if (input == node->input_values().back())
                    continue;

            const auto& new_fq_node = clone_up_fq_node(fq_node, input, copy_num);
            propagate_up(fq_node, input.get_node_shared_ptr(), copy_num);
            for (auto& consumer : input.get_target_inputs()) {
                // if FQ already exist - we don't need to generate a copy
                // also we can't insert FQ before Result node to keep output layer search logic
                const auto consumer_as_fq =
                        std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(consumer.get_node()->shared_from_this());
                const auto consumer_as_result =
                        std::dynamic_pointer_cast<ov::op::v0::Result>(consumer.get_node()->shared_from_this());
                if (consumer_as_fq == nullptr && consumer_as_result == nullptr) {
                    propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
                    consumer.replace_source_output(new_fq_node);
                }
            }
            propogate_fq(new_fq_node, copy_num);
        }
    }
}

static void propogate_fq(std::shared_ptr<ov::Node> fq_node, int& copy_num) {
    for (const auto& input : fq_node->input_values()) {
        propagate_up(fq_node, input.get_node_shared_ptr(), copy_num);
    }
    for (const auto& node_output : fq_node->outputs()) {
        for (auto& consumer : node_output.get_target_inputs()) {
            propagate_down(fq_node, consumer.get_node()->shared_from_this(), copy_num);
        }
    }
}

bool PropagateFQ::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool pass_applied = false;

    for (const std::shared_ptr<ov::Node>& node : m->get_ops()) {
        const auto fq_node = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(node);
        if (fq_node == nullptr) {
            continue;
        }

        int fq_copies_num = 0;
        for (const auto& input : fq_node->input_values()) {
            propagate_up(fq_node, input.get_node_shared_ptr(), fq_copies_num);
        }
        for (const auto& node_output : fq_node->outputs()) {
            for (auto& consumer : node_output.get_target_inputs()) {
                propagate_down(fq_node, consumer.get_node()->shared_from_this(), fq_copies_num);
            }
        }

        if (fq_copies_num != 0) {
            pass_applied = true;
        }
    }

    return pass_applied;
}

}  // namespace passes
}  // namespace vpux
