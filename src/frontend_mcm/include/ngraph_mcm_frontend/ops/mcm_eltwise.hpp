//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ngraph/op/op.hpp>
#include <memory>

class McmEltwise final : public ngraph::op::Op {
public:
    OPENVINO_OP("McmEltwise");

    enum OperationType {
        SUM = 0
    };

    McmEltwise() = default;

    McmEltwise(
            const ngraph::Output<ngraph::Node>& input0,
            const ngraph::Output<ngraph::Node>& input1,
            const OperationType operation,
            const ngraph::element::Type& type);

    void setElemType(const ngraph::element::Type& type) {
        _type = type;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    OperationType getOperationType() const { return _operation; }

private:
    ngraph::element::Type _type;
    OperationType _operation;
};

// clang-format on
