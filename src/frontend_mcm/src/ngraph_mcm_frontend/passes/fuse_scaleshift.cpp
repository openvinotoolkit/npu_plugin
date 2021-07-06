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
#include "ngraph_mcm_frontend/passes/fuse_scaleshift.hpp"
#include <ngraph/op/constant.hpp>

#include "vpux/quantization_helpers.hpp"
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/fake_quantize.hpp>
#include <vector>
#include <memory>
#include <ngraph/ops.hpp>
#include <numeric>

bool FuseScaleShift::run_on_node(std::shared_ptr<ngraph::Node> node) {
    auto convolution_add_node = std::dynamic_pointer_cast<ngraph::op::v1::Add>(node);
    if (convolution_add_node == nullptr)
        return false;

    std::shared_ptr<ngraph::Node> convolution_node = std::dynamic_pointer_cast<ngraph::op::v1::Convolution>(convolution_add_node->input_value(0).get_node_shared_ptr());
    if (convolution_node == nullptr) {
        convolution_node = std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolution>(convolution_add_node->input_value(0).get_node_shared_ptr());
        if (convolution_node == nullptr)
            return false;
    }

    const auto input_fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(convolution_node->input_value(0).get_node_shared_ptr());
    if (input_fq_node == nullptr)
        return false;

    std::vector<double> scaleshift_bias_data;
    std::vector<double> scaleshift_scale_data;

    const auto scaleshift_bias_node = std::dynamic_pointer_cast<ngraph::op::v1::Add>(input_fq_node->input_value(0).get_node_shared_ptr());
    if (scaleshift_bias_node == nullptr)
        return false;

    auto scaleshift_shifts = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleshift_bias_node->input_value(1).get_node_shared_ptr());
    if (!scaleshift_shifts) {
        scaleshift_shifts = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleshift_bias_node->input_value(0).get_node_shared_ptr());
        if (!scaleshift_shifts)
            return false;
    }
    scaleshift_bias_data = scaleshift_shifts->cast_vector<double>();

    int scaleshift_bias_to_scale_node_id = 0;
    auto scaleshift_scale_node = std::dynamic_pointer_cast<ngraph::op::v1::Multiply>(scaleshift_bias_node->input_value(0).get_node_shared_ptr());
    auto scaleshift_parameter_node = std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(scaleshift_bias_node->input_value(0).get_node_shared_ptr());
    if (!scaleshift_scale_node && !scaleshift_parameter_node) {
        scaleshift_bias_to_scale_node_id = 1;
        scaleshift_scale_node = std::dynamic_pointer_cast<ngraph::op::v1::Multiply>(scaleshift_bias_node->input_value(1).get_node_shared_ptr());
        scaleshift_parameter_node = std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(scaleshift_bias_node->input_value(1).get_node_shared_ptr());
        if (!scaleshift_scale_node && !scaleshift_parameter_node)
            return false;
    }

    int scaleshift_scale_to_input_node_id = 0;
    std::shared_ptr<ngraph::op::Constant> scaleshift_scales = nullptr;
    if (scaleshift_scale_node != nullptr) {
        scaleshift_scales = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleshift_scale_node->input_value(1).get_node_shared_ptr());
        if (!scaleshift_scales) {
            scaleshift_scale_to_input_node_id = 1;
            scaleshift_scales = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleshift_scale_node->input_value(0).get_node_shared_ptr());
            if (!scaleshift_scales)
                return false;
        }
        scaleshift_scale_data = scaleshift_scales->cast_vector<double>();
    }
    else {
        scaleshift_scale_data.push_back(1.0);
    }

    auto input_fq_node1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(1).get_node_shared_ptr());
    auto input_fq_node2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(2).get_node_shared_ptr());
    auto input_fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(3).get_node_shared_ptr());
    auto input_fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(4).get_node_shared_ptr());
    if (input_fq_node1 == nullptr || input_fq_node2 == nullptr || input_fq_node3 == nullptr || input_fq_node4 == nullptr)
        return false;

    auto weights_fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(convolution_node->input_value(1).get_node_shared_ptr());
    if (weights_fq_node == nullptr) {
        const auto convolution_weights_reshape_node = std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(convolution_node->input_value(1).get_node_shared_ptr());
        if (convolution_weights_reshape_node == nullptr)
            return false;
        weights_fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(convolution_weights_reshape_node->input_value(0).get_node_shared_ptr());
        if (weights_fq_node == nullptr)
            return false;
    }

    auto convolution_weights_node = std::dynamic_pointer_cast<ngraph::op::Constant>(weights_fq_node->input_value(0).get_node_shared_ptr());
    if (convolution_weights_node == nullptr) {
        const auto convolution_weights_convert_node = std::dynamic_pointer_cast<ngraph::op::Convert>(weights_fq_node->input_value(0).get_node_shared_ptr());
        if (convolution_weights_convert_node == nullptr)
            return false;
        convolution_weights_node = std::dynamic_pointer_cast<ngraph::op::Constant>(convolution_weights_convert_node->input_value(0).get_node_shared_ptr());
        if (convolution_weights_node == nullptr)
            return false;
    }

    auto weights_fq_node1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_fq_node->input_value(1).get_node_shared_ptr());
    auto weights_fq_node2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_fq_node->input_value(2).get_node_shared_ptr());
    auto weights_fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_fq_node->input_value(3).get_node_shared_ptr());
    auto weights_fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_fq_node->input_value(4).get_node_shared_ptr());
    if (weights_fq_node1 == nullptr || weights_fq_node2 == nullptr || weights_fq_node3 == nullptr || weights_fq_node4 == nullptr)
        return false;
    auto weights_fq_data1 = weights_fq_node1->cast_vector<double>();
    auto weights_fq_data2 = weights_fq_node2->cast_vector<double>();
    auto weights_fq_data3 = weights_fq_node3->cast_vector<double>();
    auto weights_fq_data4 = weights_fq_node4->cast_vector<double>();

    auto convolution_biases_node = std::dynamic_pointer_cast<ngraph::op::Constant>(convolution_add_node->input_value(1).get_node_shared_ptr());
    if (convolution_biases_node == nullptr)
        return false;

    int input_fq_levels = input_fq_node->get_levels();
    int weights_fq_levels = weights_fq_node->get_levels();

    auto dims = convolution_weights_node->get_output_shape(0);
    if (dims.size() != 4)
        return false;

    const size_t OC = dims[0];  // O
    const size_t IC = dims[1];  // I
    const size_t H = dims[2];   // H
    const size_t W = dims[3];   // W
    const size_t HW = H * W;
    const size_t IHW = IC * HW;

    if (scaleshift_scale_data.size() != IC) {
        if (scaleshift_scale_data.size() == 1) {
            scaleshift_scale_data.assign(IC, scaleshift_scale_data[0]);
        }
        else {
            return false;
        }
    }

    if (scaleshift_bias_data.size() != IC) {
        if (scaleshift_bias_data.size() == 1) {
            scaleshift_bias_data.assign(IC, scaleshift_bias_data[0]);
        }
        else {
            return false;
        }
    }

    // try to fuse main part in input FQ to keep accuracy in padding (ZP works like pad value here)
    double avg_scaleshift_scale = std::accumulate(scaleshift_scale_data.begin(), scaleshift_scale_data.end(), 0.0) / IC;
    double avg_scaleshift_bias = std::accumulate(scaleshift_bias_data.begin(), scaleshift_bias_data.end(), 0.0) / IC;

    // we use FQ like scaleshift (because FQ with input low/high not equal to output low/high works like scaleshift)
    // from scaleshift res = input*scale + shift
    // from FQ res = round((input - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) + output_low
    // we know that for u8 input_low=0 and input_high = 255 so after simplification
    // from FQ res = (x / 255) / (output_high - output_low) + output_low
    // from that (x / 255) / (output_high - output_low) + output_low = input*scale + shift
    // from that output_low = shift
    //           output_high = 255*scale + shift

    double input_min =   0*avg_scaleshift_scale + avg_scaleshift_bias;
    double input_max = 255*avg_scaleshift_scale + avg_scaleshift_bias;
    if (input_min > input_max) std::swap(input_min, input_max);
    if (input_min > 0) input_min = 0;
    if (input_max < 0) input_max = 0;
    double input_zp = calculateZeroPoint(input_min, input_max, input_fq_levels, ngraph::element::u8);
    double input_scale = calculateScale(input_min, input_max, input_fq_levels);
    input_min = (0 - input_zp) * input_scale;
    input_max = (255 - input_zp) * input_scale;
    if (input_scale < std::numeric_limits<double>::epsilon())
        return false;

    // if scaleshift scales closed together and to input_scale we get less accuracy drop if we don't touch weights
    auto is_scale_near = [input_scale](const double scale){return scale/input_scale < 0.99 || scale/input_scale > 1.01;};
    bool is_different_scales = std::any_of(scaleshift_scale_data.begin(), scaleshift_scale_data.end(), is_scale_near);

    replace_node_if_changed(input_fq_node1, ngraph::element::f32, 0, "_fused");
    replace_node_if_changed(input_fq_node2, ngraph::element::f32, input_fq_levels-1, "_fused");
    replace_node_if_changed(input_fq_node3, ngraph::element::f32, input_min, "_fused");
    replace_node_if_changed(input_fq_node4, ngraph::element::f32, input_max, "_fused");

    auto convolution_biases_data = (convolution_biases_node)->cast_vector<double>();
    auto convolution_weights_data = (convolution_weights_node)->cast_vector<double>();
    std::vector<float> new_weights_fq_ilo(OC);
    std::vector<float> new_weights_fq_ihi(OC);
    std::vector<float> new_weights_fq_olo(OC);
    std::vector<float> new_weights_fq_ohi(OC);
    double sumOfZeroPoints = 0;

    for (size_t oc = 0; oc < OC; ++oc) {
        double weights_fq_ilo = weights_fq_data1[std::min(weights_fq_data1.size()-1, oc)];
        double weights_fq_ihi = weights_fq_data2[std::min(weights_fq_data2.size()-1, oc)];
        double weights_fq_olo = weights_fq_data3[std::min(weights_fq_data3.size()-1, oc)];
        double weights_fq_ohi = weights_fq_data4[std::min(weights_fq_data4.size()-1, oc)];
        double weights_fq_irange = weights_fq_ihi - weights_fq_ilo;
        double weights_fq_orange = weights_fq_ohi - weights_fq_olo;
        double scaleshift_bias_acc = 0;
        double weights_min = -0.000061035156;  // fp16 closest to zero values
        double weights_max = 0.000061035156;   // used to avoid inf scales in future calculations

        for (size_t ic = 0; ic < IC; ++ic) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    const size_t idx = oc * IHW + ic * HW + h * W + w;
                    double stored_weight = convolution_weights_data[idx];
                    // dequantize weights using FQ formula
                    double real_weight = (stored_weight - weights_fq_ilo) * weights_fq_orange / weights_fq_irange + weights_fq_olo;
                    // update weights to scaleshift scale per-channel difference
                    double rescaled_weight = real_weight * scaleshift_scale_data[ic] / input_scale;
                    // update biases to scaleshift shift per-channel difference
                    double bias_modification = real_weight * (scaleshift_bias_data[ic] + input_zp * scaleshift_scale_data[ic]);
                    convolution_weights_data[idx] = rescaled_weight;
                    scaleshift_bias_acc += bias_modification;
                    // update min/max for weights FQ
                    if (weights_max < rescaled_weight) weights_max = rescaled_weight;
                    if (weights_min > rescaled_weight) weights_min = rescaled_weight;
                }
            }
        }

        new_weights_fq_ilo[oc] = weights_min;
        new_weights_fq_ihi[oc] = weights_max;
        new_weights_fq_olo[oc] = weights_min;
        new_weights_fq_ohi[oc] = weights_max;
        convolution_biases_data[oc] += scaleshift_bias_acc;
        sumOfZeroPoints += -(weights_fq_levels - 1.0) * weights_min / (weights_max - weights_min);
    }

    replace_node_if_changed(convolution_biases_node, convolution_biases_data, "");

    if (is_different_scales) {
        auto avgZeroPoints = std::round(sumOfZeroPoints / OC);
        for (size_t oc = 0; oc < OC; oc++) {
            double ol = new_weights_fq_olo[oc];
            double oh = new_weights_fq_ohi[oc];

            double zpl = oh * avgZeroPoints / (avgZeroPoints - (weights_fq_levels - 1.0));
            double zph = ol - ol * (weights_fq_levels - 1.0) / avgZeroPoints;

            ol = std::min(ol, zpl);
            oh = std::max(oh, zph);
            double scale = calculateScale(ol, oh, weights_fq_levels);
            new_weights_fq_ilo[oc] = 0;
            new_weights_fq_ihi[oc] = weights_fq_levels - 1.0;
            new_weights_fq_olo[oc] = ol;
            new_weights_fq_ohi[oc] = oh;

            for (size_t ic = 0; ic < IC; ++ic) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        const size_t idx = oc * IHW + ic * HW + h * W + w;
                        double q_weight = std::round((convolution_weights_data[idx] - ol) / scale);;
                        convolution_weights_data[idx] = clamp(q_weight, 0, weights_fq_levels - 1);
                    }
                }
            }
        }

        replace_node_if_changed(convolution_weights_node, convolution_weights_data, "");
        replace_node_if_changed(weights_fq_node1, ngraph::element::f32, new_weights_fq_ilo, "");
        replace_node_if_changed(weights_fq_node2, ngraph::element::f32, new_weights_fq_ihi, "");
        replace_node_if_changed(weights_fq_node3, ngraph::element::f32, new_weights_fq_olo, "");
        replace_node_if_changed(weights_fq_node4, ngraph::element::f32, new_weights_fq_ohi, "");
    }

    bool success1 = replace_output_update_name(scaleshift_bias_node->output(0), scaleshift_bias_node->input_value(scaleshift_bias_to_scale_node_id));
    bool success2 = scaleshift_scale_node == nullptr ||
                    replace_output_update_name(scaleshift_scale_node->output(0), scaleshift_scale_node->input_value(scaleshift_scale_to_input_node_id));
    IE_ASSERT(success1 == true && success2 == true);

    return true;
}
