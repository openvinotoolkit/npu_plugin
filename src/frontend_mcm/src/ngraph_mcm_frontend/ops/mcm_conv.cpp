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
#ifdef ENABLE_MCM_COMPILER

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
        const ngraph::Shape& output_shape,
        size_t group,
        const ngraph::element::Type& type)
            : ConvolutionIE(data, filters, strides, pads_begin, pads_end, dilations, output_shape, group), _type(type) {
    constructor_validate_and_infer_types();
}

void McmConv::validate_and_infer_types() {
    set_output_type(0, _type, m_output_shape);
}

std::shared_ptr<ngraph::Node> McmConv::copy_with_new_args(const ngraph::NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<McmConv>(
        new_args.at(0), new_args.at(1),
        m_strides, m_pads_begin, m_pads_end,
        m_dilations, m_output_shape, m_group,
        _type);
}

#endif
// clang-format on
