//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ngraph/op/op.hpp>
#include <memory>

class McmFC final : public ngraph::op::Op {
public:
    OPENVINO_OP("McmFC");

    McmFC() = default;

    McmFC(const ngraph::Output< ngraph::Node >& data,
          const ngraph::Output< ngraph::Node >& filters,
          const ngraph::Shape& output_shape,
          const ngraph::element::Type& type);

    void setElemType(const ngraph::element::Type& type) {
        _type = type;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

private:
    ngraph::element::Type _type;
    ngraph::Shape _output_shape = {};
};

// clang-format on
