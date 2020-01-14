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

#pragma once

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <ngraph/pass/pass.hpp>
#include <include/mcm/op_model.hpp>
#include <memory>

//
// Convert nGraph Function to MCM OpModel.
//

class ConvertToMcmModel final : public ngraph::pass::FunctionPass {
public:
    ConvertToMcmModel(mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) :
            _mcmModel(mcmModel), _mcmOutputsMap(mcmOutputsMap) {
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> func) override;

private:
    mv::OpModel& _mcmModel;
    NodeOutputToMcmMap& _mcmOutputsMap;
};

#endif
// clang-format on
