//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"
#include <memory>

McmFC::McmFC(const ngraph::Output<ngraph::Node>& data,
             const ngraph::Output<ngraph::Node>& filters,
             const ngraph::Shape& output_shape,
             const ngraph::element::Type& type)
    : Op({data, filters}), _type(type), _output_shape(output_shape)
{
    constructor_validate_and_infer_types();
}

void McmFC::validate_and_infer_types()
{
    set_output_type(0, _type, _output_shape);
}

std::shared_ptr<ngraph::Node> McmFC::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<McmFC>(new_args.at(0), new_args.at(1), _output_shape, _type);
}

// clang-format on
