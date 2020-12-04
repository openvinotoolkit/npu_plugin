// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <custom_layer/custom_kernel.hpp>
#include <xml_parse_utils.h>

namespace vpu {

SmallVector<std::string> deduceKernelArguments(const pugi::xml_node& node) {
    auto arguments = SmallVector<std::string>{};
    for(const auto& child : node.child("Parameters")) {
        arguments.push_back(XMLParseUtils::GetStrAttr(child, "arg-name"));
    }

    return arguments;
}

CustomKernelCpp::CustomKernelCpp(const pugi::xml_node &node, const std::string &configDir) {
    _maxShaves = XMLParseUtils::GetIntAttr(node, "max-shaves", 0);
    _kernelBinary = loadKernelBinary(node, configDir);

    processWorkSizesNode(node);

    auto bindings = processParametersNode(node);
    auto arguments = deduceKernelArguments(node);

    for (const auto& argument : arguments) {
        const auto withBindingName = [&](const BindingParameter& bind) {
            return bind.argName == argument;
        };

        auto binding = std::find_if(begin(bindings), end(bindings), withBindingName);
        IE_ASSERT(binding != bindings.end());

        _kernelBindings.push_back(*binding);
    }

    const auto isInputData = [&](const CustomKernel::BindingParameter& param) {
        return param.type == CustomParamType::Input || param.type == CustomParamType::InputBuffer ||
               param.type == CustomParamType::Data;
    };

    _inputDataCount = std::count_if(begin(_kernelBindings), end(_kernelBindings), isInputData);
}

void CustomKernelCpp::accept(CustomKernelVisitor &validator) const {
    validator.visitCpp(*this);
}

void CustomKernelCpp::processWorkSizesNode(const pugi::xml_node &node) {
    const auto workSizes = node.child("WorkSizes");

    const auto dims = XMLParseUtils::GetStrAttr(workSizes, "dim");
    std::tie(_wgDimSource, _wgDimIdx) = parseDimSource(dims);
}

} // namespace vpu
