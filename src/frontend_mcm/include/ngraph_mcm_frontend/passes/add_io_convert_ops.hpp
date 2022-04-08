//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ie_icnn_network.hpp>
#include <ngraph/pass/pass.hpp>
#include <memory>

namespace ie = InferenceEngine;

//
// Parse user input/output information from CNNNetwork and use its layout/precision for the graph.
//

class AddIOConvertOps final : public ngraph::pass::FunctionPass {
public:
    AddIOConvertOps(ie::InputsDataMap inputsInfo, ie::OutputsDataMap outputsInfo);

    bool run_on_function(std::shared_ptr<ngraph::Function> func) override;

private:
    ie::InputsDataMap _inputsInfo;
    ie::OutputsDataMap _outputsInfo;
};

// clang-format on
