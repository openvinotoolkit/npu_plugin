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

#endif
// clang-format on
