//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include <memory>

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
