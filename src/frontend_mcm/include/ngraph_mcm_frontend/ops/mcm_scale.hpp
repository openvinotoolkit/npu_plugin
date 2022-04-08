//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <memory>
#include "ngraph/op/op.hpp"

class McmScale final : public ngraph::op::Op {
public:
    OPENVINO_OP("McmScale");

    McmScale(const ngraph::Output<Node>& data_batch,
             const ngraph::Output<Node>& weights,
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
