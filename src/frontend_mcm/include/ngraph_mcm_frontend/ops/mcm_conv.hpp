//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <memory>

class McmConv final : public ngraph::op::ConvolutionIE {
public:
    OPENVINO_OP("McmConv");

    McmConv() = default;

    McmConv(
            const ngraph::Output<ngraph::Node>& data,
            const ngraph::Output<ngraph::Node>& filters,
            const ngraph::Strides& strides,
            const ngraph::CoordinateDiff& pads_begin,
            const ngraph::CoordinateDiff& pads_end,
            const ngraph::Strides& dilations,
            size_t group,
            const ngraph::element::Type& type);

    void setElemType(const ngraph::element::Type& type) {
        _type = type;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

private:
    ngraph::element::Type _type;
};

// clang-format on
