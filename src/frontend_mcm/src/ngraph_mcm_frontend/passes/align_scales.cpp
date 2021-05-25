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

#include <ie_common.h>
#include "ngraph_mcm_frontend/passes/align_scales.hpp"

#include <memory>
#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/type/element_type.hpp>
#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"
#include <ngraph/ops.hpp>

#include "ngraph_mcm_frontend/quantization_helpers.hpp"

static bool node_is_add_or_concat(std::shared_ptr<ngraph::Node> node) {
    return (std::dynamic_pointer_cast<ngraph::op::v0::Concat>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v1::Add>(node) != nullptr);
}

static std::vector<std::shared_ptr<ngraph::Node>> gather_nodes_around(std::shared_ptr<ngraph::Node> node) {
    auto result = std::vector<std::shared_ptr<ngraph::Node>>();

    for (const auto& input : node->input_values()) {
        result.push_back(input.get_node()->shared_from_this());
    }
    for (const auto& node_output : node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            result.push_back(consumer.get_node()->shared_from_this());
        }
    }

    return result;
}

static void gather_fqs(std::shared_ptr<ngraph::Node> node, std::set<std::shared_ptr<ngraph::Node>> &fqs_to_align) {
    for (const auto& input : node->input_values()) {
        auto input_node = input.get_node()->shared_from_this();
        if (std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(input_node) != nullptr) {
            if (fqs_to_align.find(input_node) == fqs_to_align.end()) {
                fqs_to_align.insert(input_node);
                auto nodes_around = gather_nodes_around(input_node);
                for (auto node_around : nodes_around) {
                    if (is_fq_agnostic(node_around)) {
                        gather_fqs(node_around, fqs_to_align);
                    }
                }
            }
        }
    }
    if (is_fq_agnostic(node)) {
        for (const auto& node_output : node->outputs()) {
            for (auto consumer : node_output.get_target_inputs()) {
                auto output_node = consumer.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(output_node) != nullptr) {
                    if (fqs_to_align.find(output_node) == fqs_to_align.end()) {
                        fqs_to_align.insert(output_node);
                        auto nodes_around = gather_nodes_around(output_node);
                        for (auto node_around : nodes_around) {
                            if (is_fq_agnostic(node_around)) {
                                gather_fqs(node_around, fqs_to_align);
                            }
                        }
                    }
                }
            }
        }
    }
}

static bool no_concat_consumers_around_fqs(std::set<std::shared_ptr<ngraph::Node>> &fqs) {
    for (auto fq : fqs) {
        auto nodes_around = gather_nodes_around(fq);
        for (auto node_around : nodes_around) {
            if (std::dynamic_pointer_cast<ngraph::op::v0::Concat>(node_around) != nullptr) {
                return false;
            }
        }
    }

    return true;
}

static bool all_fqs_have_same_io_params(std::set<std::shared_ptr<ngraph::Node>> &fqs) {
    for (auto fq_node : fqs) {
        auto fq_node1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        auto fq_node2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        auto fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
        auto fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());
        if (fq_node1 == nullptr || fq_node2 == nullptr || fq_node3 == nullptr || fq_node4 == nullptr)
            return false;

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();
        auto fq_data3 = fq_node3->cast_vector<float>();
        auto fq_data4 = fq_node4->cast_vector<float>();

        auto fq_i_counts = fq_data1.size();
        if (fq_i_counts != fq_data2.size())
            IE_THROW() << "FQ " << fq_node->get_friendly_name() << " have different input low/high parameters count";
        auto fq_o_counts = fq_data3.size();
        if (fq_o_counts != fq_data4.size())
            IE_THROW() << "FQ " << fq_node->get_friendly_name() << " have different output low/high parameters count";

        if (fq_i_counts != fq_o_counts) {
            return false;
        }

        for (size_t i = 0; i < fq_o_counts; i++) {
            if (fq_data1[i] != fq_data3[i] || fq_data2[i] != fq_data4[i]) {
                return false;
            }
        }
    }

    return true;
}

static void find_min_max(std::set<std::shared_ptr<ngraph::Node>> &fqs, float &min, float &max, float &range, int &max_levels){
    for (auto fq_node : fqs) {
        auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(fq_node);
        auto fq_node1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        auto fq_node2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        auto fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
        auto fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();
        auto fq_data3 = fq_node3->cast_vector<float>();
        auto fq_data4 = fq_node4->cast_vector<float>();

        if (max_levels < static_cast<int>(fq->get_levels())) {
            max_levels = static_cast<int>(fq->get_levels());
        }
        for (size_t c = 0; c < fq_data1.size(); c++) {
            if (min > fq_data1[c]) {
                min = fq_data1[c];
            }
            if (max < fq_data2[c]) {
                max = fq_data2[c];
            }
            if (range < fq_data2[c] - fq_data1[c]) {
                range = fq_data2[c] - fq_data1[c];
            }
        }
    }
}

static void broadcast_changes(const std::shared_ptr<ngraph::Node>& node);

static void align_fq(std::set<std::shared_ptr<ngraph::Node>> &fqs, const float min, const float max, const float range, const int max_levels) {
    auto changed = std::vector<bool>(fqs.size());
    size_t i = 0;

    for (auto fq_node : fqs) {
        auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(fq_node);
        auto fq_node1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        auto fq_node2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        auto fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
        auto fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();
        auto fq_data3 = fq_node3->cast_vector<float>();
        auto fq_data4 = fq_node4->cast_vector<float>();

        changed[i] = static_cast<int>(fq->get_levels()) != max_levels;
        fq->set_levels(max_levels);
        if (no_concat_consumers_around_fqs(fqs)) {
            // Can align for Eltwises only
            for (size_t c = 0; c < fq_data1.size(); c++) {
                double zp = calculateZeroPoint(fq_data1[c], fq_data2[c], max_levels, ngraph::element::u8);
                double scale = range / (max_levels - 1.0);
                fq_data1[c] = static_cast<float>((0.0 - zp) * scale);
                fq_data2[c] = static_cast<float>((max_levels - 1.0 - zp) * scale);
                align_zp(fq_data1[c], fq_data2[c], max_levels);
                fq_data3[c] = fq_data1[c];
                fq_data4[c] = fq_data2[c];
            }
        }
        else {
            // At least one Concat - should use Concat alignment
            for (size_t c = 0; c < fq_data1.size(); c++) {
                fq_data1[c] = min;
                fq_data2[c] = max;
                fq_data3[c] = min;
                fq_data4[c] = max;
            }
        }

        changed[i] = replace_node_if_changed(fq_node1, ngraph::element::f32, fq_data1, "_scale_aligned") || changed[i];
        changed[i] = replace_node_if_changed(fq_node2, ngraph::element::f32, fq_data2, "_scale_aligned") || changed[i];
        changed[i] = replace_node_if_changed(fq_node3, ngraph::element::f32, fq_data3, "_scale_aligned") || changed[i];
        changed[i] = replace_node_if_changed(fq_node4, ngraph::element::f32, fq_data4, "_scale_aligned") || changed[i];

        if (changed[i]) {
            if (fq->get_friendly_name().find("_scale_aligned") == std::string::npos)
                fq->set_friendly_name(fq->get_friendly_name() + "_scale_aligned");
        }
        i++;
    }

    i = 0;
    for (auto fq_node : fqs) {
        if (changed[i]) {
            broadcast_changes(fq_node);
        }
        i++;
    }
}

static void broadcast_changes(const std::shared_ptr<ngraph::Node>& node) {
    auto nodes_to_align = std::vector<std::shared_ptr<ngraph::Node>>();

    for (const auto& input : node->input_values()) {
        if (node_is_add_or_concat(input.get_node()->shared_from_this()) ||
            is_fq_agnostic(input.get_node()->shared_from_this())) {
            nodes_to_align.push_back(input.get_node()->shared_from_this());
        }
    }
    for (const auto& node_output : node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            if (node_is_add_or_concat(consumer.get_node()->shared_from_this()) ||
                is_fq_agnostic(consumer.get_node()->shared_from_this())) {
                nodes_to_align.push_back(consumer.get_node()->shared_from_this());
            }
        }
    }

    for (const auto& node_to_align : nodes_to_align) {
        std::set<std::shared_ptr<ngraph::Node>> fqs_to_align;
        gather_fqs(node_to_align, fqs_to_align);
        if (fqs_to_align.size() < 2) {
            continue;
        }
        if (!all_fqs_have_same_io_params(fqs_to_align)) {
            continue;
        }

        float min = 0;
        float max = 0;
        float range = 0;
        int max_levels = 0;
        find_min_max(fqs_to_align, min, max, range, max_levels);

        align_zp(min, max, max_levels);

        align_fq(fqs_to_align, min, max, range, max_levels);
    }
}

bool AlignScales::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (!node_is_add_or_concat(node))
        return false;

    std::set<std::shared_ptr<ngraph::Node>> fqs_to_align;
    gather_fqs(node, fqs_to_align);
    if (fqs_to_align.size() < 2) {
        return false;
    }
    if (!all_fqs_have_same_io_params(fqs_to_align)) {
        return false;
    }

    float min = 0;
    float max = 0;
    float range = 0;
    int max_levels = 0;
    find_min_max(fqs_to_align, min, max, range, max_levels);

    align_zp(min, max, max_levels);

    align_fq(fqs_to_align, min, max, range, max_levels);

    return true;
}

// clang-format on
