// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <converters.hpp>
#include <ngraph/node.hpp>
#include <ngraph/shape.hpp>
#include <ngraph_mcm_frontend/mcm_helpers.hpp>
#include <vpu/utils/simple_math.hpp>

#include "custom_layer.hpp"

namespace vpu {

namespace ie = InferenceEngine;

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

class CustomLayerParserNGraph {
    SmallVector<ngraph::Shape> _inputDescs;
    SmallVector<ngraph::Shape> _outputDescs;
    std::shared_ptr<ngraph::Node> _node;
    std::vector<mv::Data::TensorIterator> _layerInputs;
    std::map<std::string, std::string> _layerParam;
    std::unordered_map<int, mv::Data::TensorIterator> _buffers;

public:
    CustomLayerParserNGraph(std::shared_ptr<ngraph::Node>& node, std::vector<mv::Data::TensorIterator> inputs);

    std::vector<uint8_t> resolveKernelArguments(const CustomKernel& kernel,
                                                const vpu::SmallVector<uint32_t>& kernelArgs);

    std::vector<mv::TensorInfo> resolveStageOutputs(const CustomLayer& customLayer,
                                                    const std::vector<StageOutput>& stageOutputs);

    StageInfo parseKernelArguments(const SmallVector<CustomKernel::BindingParameter>& bindings);
    uint32_t parseBufferSize(const CustomKernel::BindingParameter& binding);
    void addBuffer(int port, const mv::Data::TensorIterator& bufferIt);
};

std::vector<vpu::CustomLayer::Ptr> getSuitableCustomLayers(const std::vector<vpu::CustomLayer::Ptr>& customLayers,
                                                           const std::shared_ptr<ngraph::Node>& node);

vpu::CustomLayer::Ptr findMatchingCustomLayer(const std::vector<vpu::CustomLayer::Ptr>& customLayers,
                                              const std::vector<mv::Data::TensorIterator>& inputs);

}  // namespace vpu
