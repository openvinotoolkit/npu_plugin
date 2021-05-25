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

#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include <memory>

const ngraph::NodeTypeInfo McmConv::type_info {"McmConv", 0};

McmConv::McmConv(
        const ngraph::Output<ngraph::Node>& data,
        const ngraph::Output<ngraph::Node>& filters,
        const ngraph::Strides& strides,
        const ngraph::CoordinateDiff& pads_begin,
        const ngraph::CoordinateDiff& pads_end,
        const ngraph::Strides& dilations,
        size_t group,
        const ngraph::element::Type& type)
            : ConvolutionIE(data, filters, strides, dilations, pads_begin, pads_end, group), _type(type) {
    constructor_validate_and_infer_types();
}

void McmConv::validate_and_infer_types() {
    ConvolutionIE::validate_and_infer_types();
    get_output_tensor(0).set_element_type(_type);
}

std::shared_ptr<ngraph::Node> McmConv::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<McmConv>(
        new_args.at(0), new_args.at(1),
        m_strides, m_pads_begin, m_pads_end,
        m_dilations, m_group, _type);
}

// clang-format on
