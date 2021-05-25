//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// clang-format off

#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"
#include <memory>

const ngraph::NodeTypeInfo McmFC::type_info {"McmFC", 0};

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
