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

#include "ngraph_mcm_frontend/passes/fuse_scaleshift.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/constant.hpp>

#include "ngraph_mcm_frontend/quantization_helpers.hpp"

#include <ngraph/op/fake_quantize.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>
#include <legacy/ngraph_ops/power.hpp>
#include <vector>
#include <memory>
#include <limits>
#include <ngraph_ops/convolution_ie.hpp>

bool FuseScaleShift::run_on_node(std::shared_ptr<ngraph::Node> node) {
    const auto convolution_node = std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node);
    if (convolution_node == nullptr)
        return false;

    const auto input_fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(convolution_node->input_value(0).get_node_shared_ptr());
    if (input_fq_node == nullptr)
        return false;


    const auto scaleshift_node = input_fq_node->input_value(0).get_node_shared_ptr();
    if (!scaleshift_node)
        return false;

    std::vector<double> scaleshift_scale_data;
    std::vector<double> scaleshift_bias_data;

    // in case Multiply + Add was converted to ScaleShift node
    if (scaleshift_node->get_type_info() == ngraph::op::ScaleShiftIE::type_info) {
        const auto scaleshift_scales = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleshift_node->input_value(1).get_node_shared_ptr());
        const auto scaleshift_shifts = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleshift_node->input_value(2).get_node_shared_ptr());
        if (!scaleshift_scales || !scaleshift_shifts)
            return false;

        scaleshift_scale_data = scaleshift_scales->cast_vector<double>();
        scaleshift_bias_data = scaleshift_shifts->cast_vector<double>();
    }
    else if (scaleshift_node->get_type_info() == ngraph::op::PowerIE::type_info) {
        // in some cases Multiply + Add layers might be converted in PowerIE node in graph
        auto power_node = std::dynamic_pointer_cast<ngraph::op::PowerIE>(scaleshift_node);
        // if we find PowerIE with power = 1, use it as if it was scale shift node
        if (!power_node || power_node->power != 1)
            return false;

        auto input_dims = power_node->get_input_shape(0);

        if (input_dims.size() < 2)
            return false;
        scaleshift_scale_data.assign(input_dims[1], power_node->scale);
        scaleshift_bias_data.assign(input_dims[1], power_node->shift);
    }
    else
        return false;

    auto input_fq_node1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(1).get_node_shared_ptr());
    auto input_fq_node2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(2).get_node_shared_ptr());
    auto input_fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(3).get_node_shared_ptr());
    auto input_fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(input_fq_node->input_value(4).get_node_shared_ptr());
    if (input_fq_node1 == nullptr || input_fq_node2 == nullptr || input_fq_node3 == nullptr || input_fq_node4 == nullptr)
        return false;
    auto input_fq_data1 = input_fq_node1->cast_vector<double>();
    auto input_fq_data2 = input_fq_node2->cast_vector<double>();
    auto input_fq_data3 = input_fq_node3->cast_vector<double>();
    auto input_fq_data4 = input_fq_node4->cast_vector<double>();

    const auto weights_fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(convolution_node->input_value(1).get_node_shared_ptr());
    if (weights_fq_node == nullptr)
        return false;

    const auto convolution_weights_node = std::dynamic_pointer_cast<ngraph::op::Constant>(weights_fq_node->input_value(0).get_node_shared_ptr());
    if (convolution_weights_node == nullptr)
        return false;

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

    auto convolution_biases_node = std::dynamic_pointer_cast<ngraph::op::Constant>(convolution_node->input_value(2).get_node_shared_ptr());
    if (convolution_biases_node == nullptr)
        return false;

    int input_fq_levels = input_fq_node->get_levels();
    int weights_fq_levels = weights_fq_node->get_levels();

    double input_min = 0;
    double input_max = 0;
    for (size_t ic = 0; ic < scaleshift_scale_data.size(); ++ic) {
        auto min_value = 0 * scaleshift_scale_data[ic] + scaleshift_bias_data[ic];
        auto max_value = 255 * scaleshift_scale_data[ic] + scaleshift_bias_data[ic];

        if (input_max < min_value) input_max = min_value;
        if (input_min > min_value) input_min = min_value;
        if (input_max < max_value) input_max = max_value;
        if (input_min > max_value) input_min = max_value;
    }

    for (size_t ic = 0; ic < input_fq_data1.size(); ++ic) {
        input_fq_data1[ic] = 0;
        input_fq_data2[ic] = input_fq_levels - 1;
        input_fq_data3[ic] = input_min;
        input_fq_data4[ic] = input_max;
    }

    ngraph::replace_node(input_fq_node1,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({input_fq_data1.size()}), input_fq_data1.data()));
    ngraph::replace_node(input_fq_node2,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({input_fq_data2.size()}), input_fq_data2.data()));
    ngraph::replace_node(input_fq_node3,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({input_fq_data3.size()}), input_fq_data3.data()));
    ngraph::replace_node(input_fq_node4,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({input_fq_data4.size()}), input_fq_data4.data()));

    auto dims = convolution_weights_node->get_output_shape(0);
    if (dims.size() != 4)
        return false;

    const size_t OC = dims[0];  // O
    const size_t IC = dims[1];  // I
    const size_t H = dims[2];   // H
    const size_t W = dims[3];   // W
    const size_t HW = H * W;
    const size_t IHW = IC * HW;

    double input_fq_min = input_min;
    double input_fq_max = input_max;
    double input_fq_scale = ((input_fq_max - input_fq_min) / (input_fq_levels - 1.0));
    double input_fq_zp = std::round(-(input_fq_levels - 1.0) * input_fq_min / (input_fq_max - input_fq_min));
    double input_fq_bias = -input_fq_zp;

    auto convolution_biases_data = (convolution_biases_node)->cast_vector<double>();
    auto convolution_weights_data = (convolution_weights_node)->cast_vector<double>();
    std::vector<double> new_weights_fq_ilo(OC);
    std::vector<double> new_weights_fq_ihi(OC);
    std::vector<double> new_weights_fq_olo(OC);
    std::vector<double> new_weights_fq_ohi(OC);
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
                    double real_weight = (stored_weight - weights_fq_ilo) * weights_fq_orange / weights_fq_irange + weights_fq_olo;
                    double rescaled_weight = real_weight * scaleshift_scale_data[ic] / input_fq_scale;
                    double bias_modification = real_weight * (scaleshift_bias_data[ic] - input_fq_bias * scaleshift_scale_data[ic]);
                    convolution_weights_data[idx] = rescaled_weight;
                    scaleshift_bias_acc += bias_modification;
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

    auto avgZeroPoints = std::round(sumOfZeroPoints / OC);
    for (size_t oc = 0; oc < OC; oc++) {
        double ol = new_weights_fq_olo[oc];
        double oh = new_weights_fq_ohi[oc];

        double zpl = oh * avgZeroPoints / (avgZeroPoints - (weights_fq_levels - 1.0));
        double zph = ol - ol * (weights_fq_levels - 1.0) / avgZeroPoints;

        ol = std::min(ol, zpl);
        oh = std::max(oh, zph);
        new_weights_fq_ilo[oc] = ol;
        new_weights_fq_ihi[oc] = oh;
        new_weights_fq_olo[oc] = ol;
        new_weights_fq_ohi[oc] = oh;
    }

    auto new_convolution_biases_node = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, convolution_biases_node->get_shape(), convolution_biases_data.data());
    ngraph::replace_node(convolution_biases_node, new_convolution_biases_node);

    auto new_convolution_weights_node_f64 = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, convolution_weights_node->get_shape(), convolution_weights_data.data());
    if (convolution_weights_node->get_element_type() == ngraph::element::f16) {
        auto new_convolution_weights_node_f16 = std::make_shared<ngraph::op::v0::Constant>(convolution_weights_node->get_element_type(), convolution_weights_node->get_shape(), new_convolution_weights_node_f64->cast_vector<ngraph::float16>().data());
        ngraph::replace_node(convolution_weights_node, new_convolution_weights_node_f16);
    }
    if (convolution_weights_node->get_element_type() == ngraph::element::f32) {
        auto new_convolution_weights_node_f32 = std::make_shared<ngraph::op::v0::Constant>(convolution_weights_node->get_element_type(), convolution_weights_node->get_shape(), new_convolution_weights_node_f64->cast_vector<float>().data());
        ngraph::replace_node(convolution_weights_node, new_convolution_weights_node_f32);
    }
    if (convolution_weights_node->get_element_type() == ngraph::element::f64)
        ngraph::replace_node(convolution_weights_node, new_convolution_weights_node_f64);

    ngraph::replace_node(weights_fq_node1,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({new_weights_fq_ilo.size(),1,1,1}), new_weights_fq_ilo.data()));
    ngraph::replace_node(weights_fq_node2,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({new_weights_fq_ihi.size(),1,1,1}), new_weights_fq_ihi.data()));
    ngraph::replace_node(weights_fq_node3,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({new_weights_fq_olo.size(),1,1,1}), new_weights_fq_olo.data()));
    ngraph::replace_node(weights_fq_node4,
                         std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({new_weights_fq_ohi.size(),1,1,1}), new_weights_fq_ohi.data()));

    bool success = replace_output_update_name(scaleshift_node->output(0), scaleshift_node->input_value(0));
    IE_ASSERT(success == true);

    return true;
}
