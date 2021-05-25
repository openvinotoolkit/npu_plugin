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
