//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include <memory>

McmBias::McmBias(
        const ngraph::Output<ngraph::Node>& input,
        const ngraph::Output<ngraph::Node>& bias,
        const ngraph::element::Type& type)
            : Op({input, bias}), _type(type) {
    constructor_validate_and_infer_types();
}

void McmBias::validate_and_infer_types() {
    set_output_type(0, _type, get_input_shape(0));
}

std::shared_ptr<ngraph::Node> McmBias::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<McmBias>(new_args.at(0), new_args.at(1), _type);
}

// clang-format on
