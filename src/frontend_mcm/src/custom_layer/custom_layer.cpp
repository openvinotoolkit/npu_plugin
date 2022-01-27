// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>
#include <custom_layer/custom_layer.hpp>
#include <map>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <xml_parse_utils.h>

#include <caseless.hpp>
#include <cstring>
#include <description_buffer.hpp>

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/simple_math.hpp"

namespace vpu {

namespace {

void assertExactlyOneOccurrence(const pugi::xml_node& node, const std::vector<std::string>& childs) {
    for (const auto& name : childs) {
        const auto& child = node.child(name.c_str());
        VPUX_THROW_UNLESS(!child.empty(), "Required parameter {0} is not found", name);
        VPUX_THROW_UNLESS(child.next_sibling(name.c_str()).empty(), "Found several definitions of the parameter {0}",
                          name);
    }
}

void assertOneOrMoreOccurrence(const pugi::xml_node& node, const std::vector<std::string>& childs) {
    for (const auto& name : childs) {
        const auto& child = node.child(name.c_str());
        VPUX_THROW_UNLESS(!child.empty(), "Required parameter {0} is not found", name);
    }
}

void assertZeroOrOneOccurrence(const pugi::xml_node& node, const std::vector<std::string>& childNames) {
    for (const auto& name : childNames) {
        const auto& child = node.child(name.c_str());
        VPUX_THROW_UNLESS(!child.empty() || child.next_sibling(name.c_str()).empty(),
                          "Found several definitions of the parameter {0}", name);
    }
}

void assertNoEmptyAttributes(const pugi::xml_node& customLayer) {
    const auto checkAttributes = [&customLayer](const pugi::xml_node& node) {
        for (const auto& attr : node.attributes()) {
            VPUX_THROW_UNLESS(strlen(attr.value()) != 0,
                              "Wrong custom layer XML: Custom layer {0} has node <{1}> with an empty attribute {2}",
                              customLayer.attribute("name").value(), node.name(), attr.name());
        }
    };

    checkAttributes(customLayer);

    for (const auto& child : customLayer.children()) {
        assertNoEmptyAttributes(child);
    }
}

}  // namespace

ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> CustomLayer::loadFromFile(
        const std::string& configFile, bool canBeMissed) {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_file(configFile.c_str());

    if (res.status != pugi::status_ok) {
        if (canBeMissed) {
            // Config file might not exist - like global config, for example.
            return {};
        } else {
            VPUX_THROW("Failed to load custom layer configuration file {0} : {1} at offset {2}", configFile,
                       res.description(), res.offset);
        }
    }

#ifdef _WIN32
    char path[MAX_PATH];
    auto abs_path_ptr = _fullpath(path, configFile.c_str(), MAX_PATH);
#elif defined(__linux__) || defined(__APPLE__)
    char path[PATH_MAX];
    auto abs_path_ptr = realpath(configFile.c_str(), path);
#endif

    VPUX_THROW_UNLESS(abs_path_ptr != nullptr,
                      "Failed to load custom layer configuration file {0} : can't get canonicalized absolute path",
                      configFile);

    std::string abs_file_name(path);

    // Try extracting directory from config path.
    auto dir_split_pos = abs_file_name.find_last_of("/\\");
    auto colon_pos = abs_file_name.find_first_of(':');
    auto first_slash_pos = abs_file_name.find_first_of('/');

    // If path is absolute.
    std::string dir_path;
    if (dir_split_pos != std::string::npos && (colon_pos != std::string::npos || first_slash_pos == 0)) {
        dir_path = abs_file_name.substr(0, dir_split_pos);
    } else {
        VPUX_THROW("Failed to load custom layer configuration file {0} : path is not valid", configFile);
    }

    auto out = ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>>{};
    for (auto r = xmlDoc.document_element(); r; r = r.next_sibling()) {
        auto layerPtr = std::make_shared<CustomLayer>(dir_path, r);
        out[layerPtr->_layerName].push_back(std::move(layerPtr));
    }

    return out;
}

CustomLayer::CustomLayer(std::string configDir, const pugi::xml_node& customLayer): _configDir(std::move(configDir)) {
    const auto cmp = ie::details::CaselessEq<std::string>{};
    const auto nodeName = customLayer.name();
    VPUX_THROW_UNLESS(cmp(nodeName, "CustomLayer"), "Wrong custom layer XML : Node is not CustomLayer, but {0}",
                      nodeName);

    const auto nodeType = XMLParseUtils::GetStrAttr(customLayer, "type");
    VPUX_THROW_UNLESS(cmp(nodeType, "MVCL") || cmp(nodeType, "CPP"),
                      "Wrong custom layer XML. Supported types: MVCL and CPP. Parsed type: {0}", nodeType);

    const auto version = XMLParseUtils::GetIntAttr(customLayer, "version");
    VPUX_THROW_UNLESS(version == 1, "Wrong custom layer XML : only version 1 is supported");

    _layerName = XMLParseUtils::GetStrAttr(customLayer, "name");

    assertNoEmptyAttributes(customLayer);

    assertZeroOrOneOccurrence(customLayer, {"Where"});
    const auto whereNode = customLayer.child("Where");
    for (auto where : whereNode.attributes()) {
        _whereParams[where.name()] = where.value();
    }

    assertOneOrMoreOccurrence(customLayer, {"Kernel"});
    auto kernelNodes = [&] {
        auto nodes = std::vector<pugi::xml_node>{};
        for (auto kernel = customLayer.child("Kernel"); !kernel.empty(); kernel = kernel.next_sibling("Kernel")) {
            assertExactlyOneOccurrence(kernel, {"Parameters", "WorkSizes"});
            assertOneOrMoreOccurrence(kernel, {"Source"});
            nodes.push_back(kernel);
        }
        return nodes;
    }();

    bool isCl = nodeType == "MVCL";
    auto createKernel = [&](const pugi::xml_node& node, const std::string& configDir) -> std::shared_ptr<CustomKernel> {
        if (isCl) {
            return std::make_shared<CustomKernelOcl>(node, configDir);
        }

        return std::make_shared<CustomKernelCpp>(node, configDir);
    };

    if (kernelNodes.size() == 1) {
        _kernels.emplace_back(createKernel(kernelNodes.front(), _configDir));
    } else {
        auto stageOrder = std::map<int, CustomKernel::Ptr>{};
        for (const auto& kernel : kernelNodes) {
            const auto stageAttr = kernel.attribute("stage");
            VPUX_THROW_UNLESS(stageAttr,
                              "Error while binding {0} custom layer: for multi-kernel binding, "
                              "each kernel should be provided with 'stage' attribute.",
                              _layerName);

            const auto stageNum = std::stoi(stageAttr.value());
            VPUX_THROW_UNLESS(stageOrder.find(stageNum) == stageOrder.end(),
                              "Error while binding {0} custom layer: found duplicating stage id.", _layerName);

            stageOrder.emplace(stageNum, createKernel(kernel, _configDir));
        }

        VPUX_THROW_UNLESS(stageOrder.size() > 0, "Error stage order for {0} layer is empty", _layerName);
        VPUX_THROW_UNLESS(stageOrder.begin()->first == 0, "Error while binding {0} custom layer: Stage 0 is not found.",
                          _layerName);
        VPUX_THROW_UNLESS(static_cast<size_t>(stageOrder.rbegin()->first) == stageOrder.size() - 1,
                          "Error while binding {0} custom layer: Kernels should have stage id from 0 to N.",
                          _layerName);

        for (auto& stage : stageOrder) {
            _kernels.push_back(std::move(stage.second));
        }
    }

    const auto addPorts = [](std::map<int, ie::Layout>& ports, const CustomKernel::BindingParameter& newEdge) {
        const auto layerInput = ports.find(newEdge.portIndex);
        const auto newEdgeLayout = formatToLayout(newEdge.format);
        if (layerInput == ports.end()) {
            ports.emplace(newEdge.portIndex, newEdgeLayout);
        } else if (newEdgeLayout == ie::Layout::ANY) {
            return;
        } else if (layerInput->second == ie::Layout::ANY) {
            layerInput->second = newEdgeLayout;
        }
    };

    for (const auto& kernel : _kernels) {
        for (const auto& binding : kernel->bindings()) {
            if (binding.type == CustomParamType::Input) {
                addPorts(_inputs, binding);
            }
            if (binding.type == CustomParamType::Output) {
                addPorts(_outputs, binding);
            }
        }
    }
}

bool CustomLayer::isLegalSizeRule(const std::string& rule, std::map<std::string, std::string> layerParams) {
    {
        auto sizes = std::vector<std::pair<std::string, std::string>>{
                {"b", "1"}, {"B", "1"}, {"f", "1"}, {"F", "1"}, {"y", "1"}, {"Y", "1"}, {"x", "1"}, {"X", "1"},
        };

        std::move(begin(sizes), end(sizes), inserter(layerParams, end(layerParams)));
    }

    vpux::MathExpression expr;
    expr.setVariables(layerParams);

    try {
        expr.parse(rule);
    } catch (...) {
        return false;
    }

    return true;
}

InferenceEngine::Layout CustomLayer::formatToLayout(const CustomDataFormat& format) {
    switch (format) {
    case CustomDataFormat::BFYX:
        return ie::NCHW;
    case CustomDataFormat::BYXF:
        return ie::NHWC;
    case CustomDataFormat::FYX:
        return ie::CHW;
    case CustomDataFormat::BF:
        return ie::NC;
    case CustomDataFormat::Any:
        return ie::ANY;

    case CustomDataFormat::YXF:
        break;  // Unsupported by IE
    }
    return ie::Layout::BLOCKED;
}

bool CustomLayer::meetsWhereRestrictions(const std::map<std::string, std::string>& params) const {
    const auto cmp = ie::details::CaselessEq<std::string>{};

    for (const auto& where : _whereParams) {
        const auto restrictedParam = [&](const std::pair<std::string, std::string>& param) {
            return param.first == where.first;
        };

        const auto param = std::find_if(begin(params), end(params), restrictedParam);
        if (param == params.end()) {
            return false;
        }

        const auto& restriction = where.second;
        const auto number = vpux::parseNumber<float>(param->second);

        const auto meetsRestriction = [&] {
            // compare non-number restrictions (ex. kernel="3,3")
            if (!number.hasValue()) {
                return cmp(param->second, restriction);
            } else {
                if (restriction[0] == '>' && restriction[1] == '=') {
                    const auto to_compare = std::stof(restriction.substr(2, std::string::npos));
                    return number.getValue() >= to_compare;
                } else if (restriction[0] == '<' && restriction[1] == '=') {
                    const auto to_compare = std::stof(restriction.substr(2, std::string::npos));
                    return number.getValue() <= to_compare;
                } else if (restriction[0] == '>') {
                    const auto to_compare = std::stof(restriction.substr(1, std::string::npos));
                    return number.getValue() > to_compare;
                } else if (restriction[0] == '<') {
                    const auto to_compare = std::stof(restriction.substr(1, std::string::npos));
                    return number.getValue() < to_compare;
                } else if (restriction[0] == '!' && restriction[1] == '=') {
                    const auto to_compare = std::stof(restriction.substr(2, std::string::npos));
                    return number.getValue() != to_compare;
                }
                return number.getValue() == std::stof(restriction);
            }
        }();

        if (!meetsRestriction) {
            return false;
        }
    }
    return true;
}

SizeRuleValidator::SizeRuleValidator(CustomLayer::Ptr customLayer,
                                     const std::map<std::string, std::string>& cnnLayerParams, vpux::Logger logger)
        : _customLayer(std::move(customLayer)), _cnnLayerParams(cnnLayerParams), _logger(logger) {
}

void SizeRuleValidator::visitCpp(const CustomKernelCpp&) {
    _result = true;
}

void SizeRuleValidator::visitCL(const CustomKernelOcl& kernel) {
    const auto& gws = kernel.globalGridSizeRules();
    const auto& lws = kernel.localGridSizeRules();

    const auto validSizeRule = [&](const std::string& rule) {
        return CustomLayer::isLegalSizeRule(rule, _cnnLayerParams);
    };

    const auto validGridSizes =
            std::all_of(begin(gws), end(gws), validSizeRule) && std::all_of(begin(lws), end(lws), validSizeRule);

    const decltype(lws.size()) workGroupDims = 3;
    VPUX_THROW_UNLESS(lws.size() <= workGroupDims,
                      "Failed to parse '{0}' custom layer binding list. Local work group size count "
                      "is greater than 3.",
                      _customLayer->layerName());
    VPUX_THROW_UNLESS(gws.size() <= workGroupDims,
                      "Failed to parse '{0}' custom layer binding list. Global work group size count "
                      "is greater than 3.",
                      _customLayer->layerName());

    _result = validGridSizes;
    if (!_result) {
        _logger.debug("Not suitable: Work group grid sizes are not valid");
    }
}

OperationFactory::OperationFactory(int stageIdx, mv::OpModel& modelMcm, const std::vector<uint8_t>& kernelData,
                                   const std::vector<mv::Data::TensorIterator>& stageInputs,
                                   const std::vector<mv::TensorInfo>& stageOutputs, const std::string& friendlyName)
        : _stageIdx(stageIdx),
          _modelMcm(modelMcm),
          _kernelData(kernelData),
          _stageInputs(stageInputs),
          _stageOutputs(stageOutputs),
          _friendlyName(friendlyName) {
}

void OperationFactory::visitCpp(const CustomKernelCpp& kernel) {
    const auto layerName = _friendlyName + "_CustomCpp:" + std::to_string(_stageIdx);
    _result = _modelMcm.customCpp(layerName, _stageInputs, kernel.kernelBinary(), _kernelData, _stageOutputs);
};

void OperationFactory::visitCL(const CustomKernelOcl& kernel) {
    const auto layerName = _friendlyName + "_CustomOcl:" + std::to_string(_stageIdx);
    _result = _modelMcm.customOcl(layerName, _stageInputs, kernel.kernelBinary(), _kernelData, _stageOutputs);
};

}  // namespace vpu
