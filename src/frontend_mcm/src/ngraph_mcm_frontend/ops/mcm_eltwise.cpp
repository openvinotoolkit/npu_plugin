//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"
#include <memory>

McmEltwise::McmEltwise(
        const ngraph::Output<ngraph::Node>& input0,
        const ngraph::Output<ngraph::Node>& input1,
        const OperationType operation,
        const ngraph::element::Type& type)
            : Op({input0, input1}), _type(type), _operation(operation) {
    constructor_validate_and_infer_types();
}

void McmEltwise::validate_and_infer_types() {
    set_output_type(0, _type, get_input_shape(0));
}

std::shared_ptr<ngraph::Node> McmEltwise::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<McmEltwise>(new_args.at(0), new_args.at(1), _operation, _type);
}

// clang-format on
