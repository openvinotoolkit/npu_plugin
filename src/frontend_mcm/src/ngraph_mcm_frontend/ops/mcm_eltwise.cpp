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

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"
#include <memory>

const ngraph::NodeTypeInfo McmEltwise::type_info {"McmEltwise", 0};

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

#endif // ENABLE_MCM_COMPILER
// clang-format on
