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

#include "ngraph_mcm_frontend/passes/add_io_convert_ops.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/ie_helpers.hpp"
#include <ngraph/op/convert.hpp>
#include <memory>
#include <utility>
#include <algorithm>

AddIOConvertOps::AddIOConvertOps(ie::InputsDataMap inputsInfo, ie::OutputsDataMap outputsInfo) :
        _inputsInfo(std::move(inputsInfo)), _outputsInfo(std::move(outputsInfo)) {
}

bool AddIOConvertOps::run_on_function(std::shared_ptr<ngraph::Function> func) {
    bool modified = false;

    const auto& parameters = func->get_parameters();

    for (size_t paramInd = 0; paramInd < parameters.size(); ++paramInd) {
        const auto& param = parameters[paramInd];

        const auto& inputName = param->get_friendly_name();

        const auto inputInfoIt = _inputsInfo.find(inputName);
        IE_ASSERT(inputInfoIt != _inputsInfo.end()) << "Missing information for input " << inputName;

        const auto layout = inputInfoIt->second->getLayout();
        McmOpAttrs::setOrder(cvtLayoutToMCM(layout), param);

        const auto& precision = inputInfoIt->second->getPrecision();
        const auto elemType = cvtPrecisionToElemType(precision);

        if (elemType != param->get_element_type()) {
            const auto newParam = std::make_shared<ngraph::op::Parameter>(elemType, param->get_partial_shape());
            newParam->set_friendly_name(param->get_friendly_name());
            McmOpAttrs::copy(param, newParam);

            const auto convert = std::make_shared<ngraph::op::v0::Convert>(newParam, param->get_element_type());
            convert->set_friendly_name(param->get_friendly_name() + "_convert");
            McmOpAttrs::copy(param, convert);

            func->replace_parameter(paramInd, newParam);

            for (auto& nextOpInput : param->output(0).get_target_inputs()) {
                if (nextOpInput.get_node() != convert.get()) {
                    nextOpInput.replace_source_output(convert);
                }
            }

            modified = true;
        }
    }

    for (const auto& result : func->get_results()) {
        const auto lastOp = result->input_value(0).get_node_shared_ptr();
        IE_ASSERT(lastOp != nullptr);

        const auto portInd = result->input_value(0).get_index();

        const auto& outputName = lastOp->get_friendly_name();

        const auto outputInfoIt = _outputsInfo.find(outputName);
        IE_ASSERT(outputInfoIt != _outputsInfo.end()) << "Missing information for output " << outputName;

        const auto layout = outputInfoIt->second->getLayout();
        McmOpAttrs::setOrder(cvtLayoutToMCM(layout), lastOp);

        const auto& precision = outputInfoIt->second->getPrecision();
        const auto elemType = cvtPrecisionToElemType(precision);

        if (elemType != lastOp->output(portInd).get_element_type()) {
            const auto convert = std::make_shared<ngraph::op::v0::Convert>(lastOp, elemType);
            convert->set_friendly_name(result->get_friendly_name() + "_convert");

            result->input(0).replace_source_output(convert);

            modified = true;
        }
    }

    return modified;
}

// clang-format on
