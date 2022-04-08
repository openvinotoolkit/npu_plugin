//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ngraph/op/op.hpp>
#include <memory>

class McmBias final : public ngraph::op::Op {
public:
    OPENVINO_OP("McmBias");

    McmBias() = default;

    McmBias(
            const ngraph::Output<ngraph::Node>& input,
            const ngraph::Output<ngraph::Node>& bias,
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
