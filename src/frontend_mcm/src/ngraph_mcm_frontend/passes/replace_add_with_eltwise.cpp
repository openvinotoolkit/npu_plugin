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

#include "ngraph_mcm_frontend/passes/replace_add_with_eltwise.hpp"

#include <memory>
#include "ngraph/op/add.hpp"
#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"


bool ReplaceAddWithMcmEltwise::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (const auto add = std::dynamic_pointer_cast<ngraph::op::v1::Add>(node))
    {
        const auto mcmEltwise = std::make_shared<McmEltwise>(
            add->input_value(0),
            add->input_value(1),
            McmEltwise::OperationType::SUM,
            add->get_output_element_type(0));

        ngraph::replace_node(add, mcmEltwise);
        return true;
    }
    return false;
}
#endif  // ENABLE_MCM_COMPILER
// clang-format on
