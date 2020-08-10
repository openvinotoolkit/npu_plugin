// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <converters.hpp>
#include <vpu/utils/simple_math.hpp>

namespace vpu {

class CustomLayerParser {
    struct StageOutput {
        bool isBuffer;
        int bufferSize;
        int portIndex;
        std::string argName;

        StageOutput(bool is_buffer, int buffer_size, int port_index, std::string arg_name)
            : isBuffer(is_buffer),
              bufferSize(buffer_size),
              portIndex(port_index),
              argName{std::move(arg_name)} {}
    };

    SmallVector<ie::TensorDesc> inputDescs;
    SmallVector<ie::TensorDesc> outputDescs;

    InferenceEngine::CNNLayerPtr cnnLayer;
    McmNodeVector layerInputs;
    std::unordered_map<int, mv::Data::TensorIterator> buffers;

private:
    static SmallVector<int> calcSizesFromParams(const ie::TensorDesc& desc,
        const SmallVector<std::string>& bufferSizeRules,
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
        std::transform(begin(bufferSizeRules), end(bufferSizeRules),
            std::back_inserter(parsedSizes), parseSizeRule);

        return parsedSizes;
    }

public:
    CustomLayerParser(InferenceEngine::CNNLayerPtr layer, McmNodeVector inputs)
        : cnnLayer{std::move(layer)}, layerInputs{std::move(inputs)} {
        inputDescs.reserve(inputs.size());
        for (const auto& input : inputs) {
            inputDescs.push_back(input->desc());
        }

        outputDescs.reserve(cnnLayer->outData.size());
        for (const auto& outData : cnnLayer->outData) {
            outputDescs.push_back(outData->getTensorDesc());
        }
    };

    std::vector<uint8_t> resolveKernelArguments(
        const CustomKernel& kernel, const vpu::SmallVector<uint32_t>& kernelArgs) {
        const auto workGroupDims = 3;

        const auto& wgDimSource =
            (kernel.dimSource() == CustomDimSource::Input) ? inputDescs : outputDescs;
        const auto& wgDataDesc = wgDimSource.at(kernel.dimSourceIndex());

        auto lwgs = calcSizesFromParams(wgDataDesc, kernel.localGridSizeRules(), cnnLayer->params);
        for (int i = lwgs.size(); i < workGroupDims; i++) {
            lwgs.push_back(1);
        }

        auto gwgs = calcSizesFromParams(wgDataDesc, kernel.globalGridSizeRules(), cnnLayer->params);
        for (int i = gwgs.size(); i < workGroupDims; i++) {
            gwgs.push_back(1);
        }

        const auto globalOffset = std::array<uint32_t, workGroupDims>{0};

        auto kernelParams = std::vector<uint32_t>{};
        kernelParams.reserve(workGroupDims * 3 + 2 + kernelArgs.size());

        std::copy(begin(lwgs), end(lwgs), back_inserter(kernelParams));
        for (size_t i = 0; i < lwgs.size(); i++) {
            IE_ASSERT(gwgs[i] % lwgs[i] == 0);
            kernelParams.push_back(gwgs[i] / lwgs[i]);
        }
        std::copy(globalOffset.begin(), globalOffset.end(), std::back_inserter(kernelParams));
        kernelParams.push_back(workGroupDims);
        kernelParams.push_back(kernel.kernelId());

        std::copy(kernelArgs.begin(), kernelArgs.end(), std::back_inserter(kernelParams));

        auto kernelData = std::vector<uint8_t>(kernelParams.size() * sizeof(uint32_t));
        std::copy(kernelParams.begin(), kernelParams.end(), reinterpret_cast<uint32_t*>(kernelData.data()));

        return kernelData;
    }

    std::vector<mv::TensorInfo> resolveStageOutputs(const CustomKernel& kernel,
        const CustomLayer& customLayer, const std::vector<StageOutput>& stageOutputs) {
        std::vector<mv::TensorInfo> kernelOutputs;
        for (const auto& output : stageOutputs) {
            if (output.isBuffer) {
                kernelOutputs.emplace_back(
                    mv::Shape{static_cast<uint32_t>(output.bufferSize), 1, 1, 1},
                    mv::DType{"UInt8"}, mv::Order::getZMajorID(4));
            } else {
                const auto& desc = outputDescs.at(output.portIndex);
                VPU_THROW_UNLESS(
                    desc.getDims().size() <= 4, "Custom layer does not support tensors greater 4D");
                auto shape = sizeVectorToShape(desc.getDims());
                // Propagate shape to 4D, adding 1's on major dimensions
                shape = mv::Shape::augment_major(shape, 4);

                const auto layerOutputs = customLayer.outputs();
                const auto outputLayoutIt = layerOutputs.find(output.portIndex);
                VPU_THROW_UNLESS(outputLayoutIt != layerOutputs.end(),
                    "Failed to parse custom layer '%s'. "
                    "Couldn't find output tensor with port-index=%l ",
                    customLayer.layerName(), output.portIndex);

                auto order = layoutToOrder(outputLayoutIt->second);
                // 4D tensor can be only in two layouts: NHWC (default) or NCHW.
                if (order != mv::Order::getColMajorID(4)) {
                    order = mv::Order::getZMajorID(4);
                }
                const auto type = [&] {
                    const auto withBindingName = [&](const CustomKernel::Argument& arg) {
                        return arg.name == output.argName;
                    };

                    auto argument = std::find_if(
                        begin(kernel.arguments()), end(kernel.arguments()), withBindingName);
                    IE_ASSERT(argument != kernel.arguments().end());

                    if (argument->underlyingTypeSize == 1) return mv::DType{"UInt8"};
                    if (argument->underlyingTypeSize == 2) return mv::DType{"Float16"};

                    VPU_THROW_EXCEPTION
                        << "Custom layer output parameter '" << output.argName
                        << "' has unsupported output data type with "
                        << "underlying type size = " << argument->underlyingTypeSize;
                }();

                // setting type as `Default` to replace it with input[0]'s DType inside MCM
                // using actual type is failing to compile with YoloV2 IR
                kernelOutputs.emplace_back(shape, mv::DType{"Default"}, order);
            }
        }

        return kernelOutputs;
    }

    struct StageInfo {
        std::vector<uint32_t> arguments;
        std::vector<mv::Data::TensorIterator> inputs;
        std::vector<StageOutput> outputs;
    };

    StageInfo parseKernelArguments(const std::vector<CustomKernel::BindingParameter>& bindings) {
        const auto floatAsInt = [](const float f) {
            uint32_t i;
            memcpy(&i, &f, 4);
            return i;
        };

        StageInfo stage;

        for (const auto& binding : bindings) {
            switch (binding.type) {
            case CustomParamType::InputBuffer: {
                const auto bufferIt = buffers.find(binding.portIndex);
                VPU_THROW_UNLESS(bufferIt != buffers.end(),
                    "Unable to deduce parameter '%s' for '%s' layer. "
                    "There is no output_buffer with port-index=%d defined.",
                    binding.argName, cnnLayer->type, binding.portIndex);

                stage.inputs.push_back(bufferIt->second);
                stage.arguments.push_back(stage.inputs.size() - 1);
                break;
            }
            case CustomParamType::OutputBuffer: {
                VPU_THROW_UNLESS(buffers.find(binding.portIndex) == buffers.end(),
                    "Unable to deduce parameter '%s' for '%s' layer. "
                    "Can't add output_buffer with port-index=%d. "
                    "Buffer with that index already exists.",
                    binding.argName, cnnLayer->type, binding.portIndex);

                const int bufferSize = parseBufferSize(binding);
                stage.outputs.emplace_back(true, bufferSize, binding.portIndex, binding.argName);
                stage.arguments.push_back(stage.outputs.size() - 1);
                break;
            }
            case CustomParamType::Data:
            case CustomParamType::Input: {
                VPU_THROW_UNLESS((uint32_t)binding.portIndex < layerInputs.size(),
                    "Unable to deduce parameter '%s' for '%s' layer. "
                    "Can't find layer input with port-index=%d.",
                    binding.argName, cnnLayer->type, binding.portIndex);

                stage.inputs.push_back(layerInputs.at(binding.portIndex)->getMcmNode());
                stage.arguments.push_back(stage.inputs.size() - 1);
                break;
            }
            case CustomParamType::Output: {
                stage.outputs.emplace_back(false, 0, binding.portIndex, binding.argName);
                stage.arguments.push_back(stage.outputs.size() - 1);
                break;
            }
            case CustomParamType::Int:
            case CustomParamType::Float: {
                const auto cnnParam = cnnLayer->params.find(binding.irSource);
                if (cnnParam != cnnLayer->params.end()) {
                    // parse cnnLayer param
                    const auto param = [&]() -> std::string {
                        if (binding.portIndex < 0) {
                            VPU_THROW_UNLESS(parseNumber<int>(cnnParam->second).hasValue(),
                                "Unable to deduce parameter '%s' for '%s' layer. Without "
                                "port-index set, only viable "
                                "size value is a whole integer number.",
                                binding.argName, cnnLayer->type);
                            return cnnParam->second;
                        }

                        VPU_THROW_UNLESS(cnnParam->second.find(',') != std::string::npos,
                            "Error while parsing CNNetwork parameter '%s' for '%s' layer: "
                            "port-index=%d is set, "
                            "but parameter is neither a tensor, nor an array type.",
                            cnnParam->first, cnnLayer->type, binding.portIndex);

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
                            binding.argName, cnnLayer->type, cnnLayer->name, binding.irSource);
                        stage.arguments.push_back(val.get());
                    } else {
                        const auto val = parseNumber<float>(param);
                        VPU_THROW_UNLESS(val.hasValue(),
                            "Unable to deduce parameter '%s' for '%s' layer. "
                            "Name is: '%s', parameter is: '%s'",
                            binding.argName, cnnLayer->type, cnnLayer->name, binding.irSource);
                        stage.arguments.push_back(floatAsInt(val.get()));
                    }
                    // if not cnnLayer param, check if it is 'I.X' format param
                } else if (binding.irSource[1] == '.' &&
                           (binding.irSource[0] == 'I' || binding.irSource[0] == 'O')) {
                    VPU_THROW_UNLESS(binding.irSource.length() == 3,
                        "Unable to deduce parameter '%s' for '%s' layer."
                        "Wrong source format",
                        binding.argName, cnnLayer->type);

                    const auto origData = [&] {
                        if (binding.irSource[0] == 'I') {
                            return cnnLayer->insData[binding.portIndex].lock();
                        }
                        return cnnLayer->outData[binding.portIndex];
                    }();
                    IE_ASSERT(origData != nullptr);

                    const auto dimLetter = toupper(binding.irSource[2]);
                    auto dims = origData->getDims();
                    const auto dimPosition = [&] {
                        if (dims.size() == 4) {
                            return std::string{"BFYX"}.find(dimLetter);
                        } else if (dims.size() == 3) {
                            return std::string{"FYX"}.find(dimLetter);
                        } else if (dims.size() == 2) {
                            return std::string{"BF"}.find(dimLetter);
                        } else {
                            return std::string::npos;
                        }
                    }();

                    VPU_THROW_UNLESS(dimPosition != std::string::npos,
                        "Unable to deduce parameter '%s' for '%s' layer."
                        "Failed to parse source dimension from provided string '%s'",
                        binding.argName, cnnLayer->type, binding.irSource);

                    auto dimValue = dims.at(dimPosition);
                    stage.arguments.push_back(static_cast<uint32_t>(dimValue));
                } else {
                    VPU_THROW_UNLESS(binding.portIndex < 0,
                        "Unable to deduce parameter '%s' for '%s' layer: port-index=%d is set, "
                        "but parameter is neither a tensor, nor an array type.",
                        binding.argName, cnnLayer->type, binding.portIndex);

                    const auto val = [&] {
                        if (binding.type == CustomParamType::Int) {
                            return parseNumber<int>(binding.irSource);
                        } else {
                            return parseNumber<float>(binding.irSource);
                        }
                    }();

                    VPU_THROW_UNLESS(val.hasValue(),
                        "Unable to deduce parameter '%s' for '%s' layer. "
                        "Name is: '%s', parameter is: '%s'",
                        binding.argName, cnnLayer->type, cnnLayer->name, binding.irSource);

                    const auto number =
                        binding.type == CustomParamType::Int ? val.get() : floatAsInt(val.get());
                    stage.arguments.push_back(number);
                }
                break;
            }
            case CustomParamType::LocalData: {
                stage.arguments.push_back(parseBufferSize(binding));
                break;
            }
            }
        }

        return stage;
    }

    uint32_t parseBufferSize(const CustomKernel::BindingParameter& binding) {
        const auto& source = binding.dimSource == CustomDimSource::Input ? inputDescs : outputDescs;
        const auto& desc = source[binding.dimIdx];
        const auto sizes = calcSizesFromParams(desc, {binding.bufferSizeRule}, cnnLayer->params);
        return sizes[0];
    }

    void addBuffer(int port, const mv::Data::TensorIterator& bufferIt) {
        buffers.emplace(port, bufferIt);
    }
};

}  // namespace vpu
