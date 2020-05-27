#pragma once
#include <ie_common.h>
#include <ie_layouts.h>

#include <array>
#include <cstring>
#include <custom_layer/custom_layer.hpp>
#include <vector>
#include <vpu/utils/simple_math.hpp>

#include "include/mcm/compiler/compilation_unit.hpp"

using namespace InferenceEngine;

using half = ushort;

namespace vpu {

static SmallVector<int> calcSizesFromParams(const ie::TensorDesc& desc, const SmallVector<std::string>& bufferSizeRules,
    std::map<std::string, std::string> layerParams) {
    const auto& dims = desc.getDims();
    const auto B = std::to_string(dims[0]);
    const auto F = std::to_string(dims[1]);
    const auto Y = std::to_string(dims[2]);
    const auto X = std::to_string(dims[3]);

    auto sizes = std::vector<std::pair<std::string, std::string>>{
        {"b", B}, {"B", B},
        {"f", F}, {"F", F},
        {"y", Y}, {"Y", Y},
        {"x", X}, {"X", X},
    };

    std::move(begin(sizes), end(sizes), inserter(layerParams, end(layerParams)));

    MathExpression expr;
    expr.setVariables(layerParams);

    const auto parseSizeRule = [&expr](const std::string& rule) {
        expr.parse(rule);
        return expr.evaluate();
    };

    auto parsedSizes = SmallVector<int>{};
    parsedSizes.reserve(bufferSizeRules.size());
    std::transform(begin(bufferSizeRules), end(bufferSizeRules), std::back_inserter(parsedSizes), parseSizeRule);

    return parsedSizes;
}

static uint32_t parseKernelArgument(const CustomKernel::KernelParam& binding, const CNNLayerPtr& layer,
    const SmallVector<TensorDesc>& inputDescs, const SmallVector<TensorDesc>& outputDescs) {

    const auto floatAsInt = [](const float f) {
        uint32_t i;
        memcpy(&i, &f, 4);
        return i;
    };

    switch (binding.type) {
        case CustomParamType::InputBuffer:
        case CustomParamType::OutputBuffer:
        case CustomParamType::Data: {
            VPU_THROW_EXCEPTION << "Unsupported parameter for KMB";
        }
        case CustomParamType::Input:
        case CustomParamType::Output: {
            return binding.portIndex;
        }
        case CustomParamType::Int:
        case CustomParamType::Float: {
            const auto cnnParam = layer->params.find(binding.irSource);
            if (cnnParam != layer->params.end()) {
                // parse cnnLayer param
                const auto param = [&]() -> std::string {
                    if (binding.portIndex < 0) {
                        return cnnParam->second;
                    }

                    VPU_THROW_UNLESS(cnnParam->second.find(',') != std::string::npos,
                        "Error while parsing CNNetwork parameter '%s' for '%s' layer: port-index=%d is set, "
                        "but parameter is neither a tensor, nor an array type.",
                        cnnParam->first, layer->type, binding.portIndex);

                    std::string value;
                    std::stringstream parameterStream{cnnParam->second};
                    for (int i = 0; i <= binding.portIndex; i++) {
                        getline(parameterStream, value, ',');
                    }
                    return value;
                }();
                if (binding.type == CustomParamType::Int) {
                    const auto val = parseNumber<int>(param);
                    VPU_THROW_UNLESS(val.hasValue(),
                        "Unable to deduce parameter '%s' for '%s' layer. "
                        "Name is: '%s', parameter is: '%s'",
                        binding.argName, layer->type, layer->name, binding.irSource);
                    return val.get();
                } else {
                    const auto val = parseNumber<float>(param);
                    VPU_THROW_UNLESS(val.hasValue(),
                        "Unable to deduce parameter '%s' for '%s' layer. "
                        "Name is: '%s', parameter is: '%s'",
                        binding.argName, layer->type, layer->name, binding.irSource);
                    return floatAsInt(val.get());
                }
            } else {
                // if not cnnLayer param, check if it is 'I.X' format param
                auto pos = binding.irSource.find_first_of('.');
                if (pos != std::string::npos && (binding.irSource[0] == 'I' || binding.irSource[0] == 'O')) {
                    auto blob = binding.irSource.substr(0, pos);
                    auto dim = binding.irSource.substr(pos + 1, std::string::npos);

                    VPU_THROW_UNLESS(dim.length() == 1, "Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'",
                        binding.argName, layer->type, layer->name);

                    char dimLetter = dim[0];

                    ie::DataPtr origData;
                    if (blob == "I") {
                        origData = layer->insData[binding.portIndex].lock();
                    } else {
                        origData = layer->outData[binding.portIndex];
                    }
                    IE_ASSERT(origData != nullptr);

                    auto dims = origData->getDims();
                    int ndims = dims.size();

                    if (ndims > 4) {
                        VPU_THROW_UNLESS(dim.length() == 1,
                            "Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'", binding.argName,
                            layer->type, layer->name);
                    }
                    const std::map<char, int> vars = {
                        {'b', 0}, {'B', 0},
                        {'f', 1}, {'F', 1},
                        {'y', 2}, {'Y', 2},
                        {'x', 3}, {'X', 3},
                    };

                    auto var = vars.find(dimLetter);
                    if (var != vars.end()) {
                        auto res = dims.at(var->second - 4 + ndims);

                        return static_cast<uint32_t>(res);
                    } else {
                        VPU_THROW_FORMAT("Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'",
                            binding.argName, layer->type, layer->name);
                    }

                    break;
                } else {
                    VPU_THROW_UNLESS(binding.portIndex < 0,
                        "Unable to deduce parameter '%s' for '%s' layer: port-index=%d is set, "
                        "but parameter is neither a tensor, nor an array type.",
                        binding.argName, layer->type, binding.portIndex);
                    try {
                        if (binding.type == CustomParamType::Int) {
                            return std::stoi(binding.irSource);
                        } else {
                            return floatAsInt(std::stof(binding.irSource));
                        }
                    } catch (const std::invalid_argument&) {
                        VPU_THROW_FORMAT("Unable to deduce parameter '%s' for '%s' layer. "
                                         "Name is: '%s', parameter is: '%s'",
                            binding.argName, layer->type, layer->name, binding.irSource);
                    }
                }
            }
        }
        case CustomParamType::LocalData: {
            const auto& source = binding.dimSource == CustomDimSource::Input ? inputDescs : outputDescs;
            const auto& desc = source[binding.dimIdx];
            const auto sizes = calcSizesFromParams(desc, {binding.bufferSizeRule}, layer->params);
            return sizes[0];
        }
        default:
            VPU_THROW_FORMAT("Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'", binding.argName,
                layer->type, layer->name);
        }
}

}  // namespace vpu