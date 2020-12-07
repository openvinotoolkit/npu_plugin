// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <converters.hpp>
#include <custom_layer/custom_kernel.hpp>
#include <frontend_mcm.hpp>
#include <vpu/utils/simple_math.hpp>

#include "custom_layer.hpp"

namespace vpu {

namespace ie = InferenceEngine;

class CustomLayerParser {
public:
    struct StageOutput {
        bool isBuffer;
        int bufferSize;
        int portIndex;
        std::string argName;

        StageOutput(bool is_buffer, int buffer_size, int port_index, std::string arg_name)
                : isBuffer(is_buffer), bufferSize(buffer_size), portIndex(port_index), argName{std::move(arg_name)} {
        }
    };

    struct StageInfo {
        std::vector<uint32_t> arguments;
        std::vector<mv::Data::TensorIterator> inputs;
        std::vector<StageOutput> outputs;
    };

public:
    CustomLayerParser(InferenceEngine::CNNLayerPtr layer, McmNodeVector inputs);

    std::vector<uint8_t> resolveKernelArguments(const CustomKernel& kernel,
                                                const vpu::SmallVector<uint32_t>& kernelArgs);
    std::vector<mv::TensorInfo> resolveStageOutputs(const CustomLayer& customLayer,
                                                    const std::vector<StageOutput>& stageOutputs);

    StageInfo parseKernelArguments(const SmallVector<CustomKernel::BindingParameter>& bindings);
    uint32_t parseBufferSize(const CustomKernel::BindingParameter& binding);

    void addBuffer(int port, const mv::Data::TensorIterator& bufferIt);

private:
    SmallVector<ie::TensorDesc> inputDescs;
    SmallVector<ie::TensorDesc> outputDescs;

    InferenceEngine::CNNLayerPtr cnnLayer;
    McmNodeVector layerInputs;
    std::unordered_map<int, mv::Data::TensorIterator> buffers;
};

}  // namespace vpu
