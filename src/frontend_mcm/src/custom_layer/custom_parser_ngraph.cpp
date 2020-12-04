#include <debug.h>

#include <custom_layer/custom_parser_ngraph.hpp>

class paramVisitor : public ngraph::AttributeVisitor {
    std::map<std::string, std::string> layerParam;

public:
    std::map<std::string, std::string> GetMap() const {
        return layerParam;
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        (void)name;
        (void)adapter;
        throw std::logic_error("Adapter is not handled");
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        layerParam[name] = adapter.get() ? "1" : "0";
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        layerParam[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        layerParam[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        layerParam[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        auto data = adapter.get();
        layerParam[name] = InferenceEngine::details::joinVec(data);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        auto data = adapter.get();
        layerParam[name] = InferenceEngine::details::joinVec(data);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        auto data = adapter.get();
        layerParam[name] = InferenceEngine::details::joinVec(data);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> data = adapter.get();
        for (auto& str : data) {
            std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
        }

        std::stringstream ss;
        std::copy(data.begin(), data.end(), std::ostream_iterator<std::string>(ss, ","));
        layerParam[name] = ss.str();
    }
};

namespace vpu {

static SmallVector<int> calcSizesFromParams(const std::vector<size_t>& dims,
                                            const SmallVector<std::string>& bufferSizeRules,
                                            std::map<std::string, std::string> layerParams) {
    const auto B = std::to_string(dims[0]);
    const auto F = std::to_string(dims[1]);
    const auto Y = std::to_string(dims[2]);
    const auto X = std::to_string(dims[3]);

    auto sizes = std::vector<std::pair<std::string, std::string>>{
            {"b", B}, {"B", B}, {"f", F}, {"F", F}, {"y", Y}, {"Y", Y}, {"x", X}, {"X", X},
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

class CustomKernelParserNGraph : public CustomKernelVisitor {
public:
    CustomKernelParserNGraph(std::vector<uint32_t>& kernelParams,
                             const std::map<std::string, std::string>& cnnLayerParams,
                             const SmallVector<ngraph::Shape>& inputDescs,
                             const SmallVector<ngraph::Shape>& outputDescs,
                             const vpu::SmallVector<uint32_t>& kernelArgs) :
        _kernelParams(kernelParams), _cnnLayerParams(cnnLayerParams),
        _inputDescs(inputDescs), _outputDescs(outputDescs),
        _kernelArgs(kernelArgs){}

    void visitCpp(const CustomKernelCpp&) override {
        _kernelParams.push_back(_kernelArgs.size());
    }

    void visitCL(const CustomKernelOcl& kernel) override {
        const auto workGroupDims = 3;

        const auto& wgDimSource = (kernel.dimSource() == CustomDimSource::Input) ? _inputDescs : _outputDescs;
        const auto& wgDataDesc = wgDimSource.at(kernel.dimSourceIndex());

        auto lwgs = calcSizesFromParams(wgDataDesc, kernel.localGridSizeRules(), _cnnLayerParams);
        for (auto i = lwgs.size(); i < workGroupDims; i++) {
            lwgs.push_back(1);
        }

        auto gwgs = calcSizesFromParams(wgDataDesc, kernel.globalGridSizeRules(), _cnnLayerParams);
        for (auto i = gwgs.size(); i < workGroupDims; i++) {
            gwgs.push_back(1);
        }

        const auto globalOffset = std::array<uint32_t, workGroupDims>{0};

        std::copy(begin(lwgs), end(lwgs), back_inserter(_kernelParams));
        for (decltype(lwgs.size()) i = 0; i < lwgs.size(); i++) {
            IE_ASSERT(gwgs[i] % lwgs[i] == 0);
            _kernelParams.push_back(gwgs[i] / lwgs[i]);
        }
        std::copy(globalOffset.begin(), globalOffset.end(), std::back_inserter(_kernelParams));
        _kernelParams.push_back(workGroupDims);
        _kernelParams.push_back(kernel.kernelId());
    }

private:
    std::vector<uint32_t>& _kernelParams;
    const std::map<std::string, std::string>& _cnnLayerParams;
    const SmallVector<ngraph::Shape>& _inputDescs;
    const SmallVector<ngraph::Shape>& _outputDescs;
    const vpu::SmallVector<uint32_t>& _kernelArgs;
};

CustomLayerParserNGraph::CustomLayerParserNGraph(std::shared_ptr<ngraph::Node>& node,
                                                 std::vector<mv::Data::TensorIterator> inputs)
    : _node(node), _layerInputs(std::move(inputs)) {
    paramVisitor visitor;
    node->visit_attributes(visitor);
    _layerParam = visitor.GetMap();

    _inputDescs.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); i++) {
        _inputDescs.push_back(node->get_input_shape(i));
    }

    _outputDescs.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_output_size(); i++) {
        _outputDescs.push_back(node->get_output_shape(i));
    }
};

std::vector<vpu::CustomLayer::Ptr> getSuitableCustomLayers(const std::vector<vpu::CustomLayer::Ptr>& customLayers,
                                                           const std::shared_ptr<ngraph::Node>& node) {
    const auto isSuitableLayer = [&](const vpu::CustomLayer::Ptr& customLayer) {
        paramVisitor visitor;
        node->visit_attributes(visitor);
        auto layerParams = visitor.GetMap();

        if (!customLayer->meetsWhereRestrictions(layerParams)) {
            return false;
        }

        SizeRuleValidator validator{customLayer, layerParams};
        for (const auto& kernel : customLayer->kernels()) {
            kernel->accept(validator);
            if (!validator.result()) {
                return false;
            }
        }

        return true;
    };

    auto suitableCustomLayers = vpu::SmallVector<vpu::CustomLayer::Ptr>{};

    std::copy_if(begin(customLayers), end(customLayers), back_inserter(suitableCustomLayers), isSuitableLayer);

    return suitableCustomLayers;
}

vpu::CustomLayer::Ptr findMatchingCustomLayer(const std::vector<vpu::CustomLayer::Ptr>& customLayers,
                                              const std::vector<mv::Data::TensorIterator>& inputs) {
    VPU_THROW_UNLESS(customLayers.size(), "Trying to use custom layer parser but custom layers were not found.");

    const auto findMismatchedClEdge = [&](const vpu::SmallVector<mv::Order>& cnnEdges,
                                          const std::map<int, InferenceEngine::Layout>& clEdges) {
        for (auto clEdge = begin(clEdges); clEdge != end(clEdges); ++clEdge) {
            int port = clEdge->first;
            VPU_THROW_UNLESS(port < (int)cnnEdges.size(),
                             "Can't bind custom layer edge with port '%s' to CNNNetwork layer", port);

            const auto clFormat = clEdge->second;
            const auto cnnFormat = cnnEdges[port];
            if (cnnFormat != layoutToOrder(clFormat) && clFormat != InferenceEngine::Layout::ANY) {
                return clEdge;
            }
        }
        return end(clEdges);
    };

    const auto mcmInputs = [&] {
        auto mcmInputs = vpu::SmallVector<mv::Order>{};
        mcmInputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            const auto layout = input->getOrder();
            mcmInputs.push_back(layout);
        }
        return mcmInputs;
    }();

    for (const auto& customLayer : customLayers) {
        const auto clInputs = customLayer->inputs();

        if (findMismatchedClEdge(mcmInputs, clInputs) == end(clInputs)) {
            return customLayer;
        }
    }

    // If we haven't returned yet, layouts have not matched (error)
    const auto& firstLayer = customLayers.front();
    const auto& layerName = firstLayer->layerName();
    const auto& clInputs = firstLayer->inputs();

    const auto mismatch = findMismatchedClEdge(mcmInputs, clInputs);
    const auto port = mismatch->first;
    const auto& mcmOrder = mcmInputs.at(port);
    const auto clOrder = layoutToOrder(mismatch->second);

    VPU_THROW_FORMAT("Failed to bind '%l' custom layer. MCM compiler expected input with port '%l' to have '%l' "
                     "layout, but custom layer has '%l' layout instead.",
                     layerName, port, mcmOrder.toString(), clOrder.toString());

    return nullptr;
}

std::vector<uint8_t> CustomLayerParserNGraph::resolveKernelArguments(const CustomKernel& kernel,
                                                                     const vpu::SmallVector<uint32_t>& kernelArgs) {
    auto kernelParams = std::vector<uint32_t>{};

    CustomKernelParserNGraph kernelParser{kernelParams, _layerParam,
                                          _inputDescs, _outputDescs, kernelArgs};
    kernel.accept(kernelParser);

    std::copy(kernelArgs.begin(), kernelArgs.end(), std::back_inserter(kernelParams));

    auto kernelData = std::vector<uint8_t>(kernelParams.size() * sizeof(uint32_t));
    std::copy(kernelParams.begin(), kernelParams.end(), reinterpret_cast<uint32_t*>(kernelData.data()));

    return kernelData;
}

std::vector<mv::TensorInfo> CustomLayerParserNGraph::resolveStageOutputs(const CustomLayer& customLayer,
                                                                         const std::vector<StageOutput>& stageOutputs) {
    std::vector<mv::TensorInfo> kernelOutputs;
    for (const auto& output : stageOutputs) {
        if (output.isBuffer) {
            kernelOutputs.emplace_back(mv::Shape{static_cast<uint32_t>(output.bufferSize), 1, 1, 1}, mv::DType{"UInt8"},
                                       mv::Order::getZMajorID(4));
        } else {
            const auto& desc = _outputDescs.at(output.portIndex);
            VPU_THROW_UNLESS(desc.size() <= 4, "Custom layer does not support tensors greater 4D");
            auto shape = sizeVectorToShape(desc);
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

            // setting type as `Default` to replace it with input[0]'s DType
            // inside MCM using actual type is failing to compile with YoloV2 IR
            kernelOutputs.emplace_back(shape, mv::DType{"Default"}, order);
        }
    }

    return kernelOutputs;
}

StageInfo CustomLayerParserNGraph::parseKernelArguments(const SmallVector<CustomKernel::BindingParameter>& bindings) {
    const auto floatAsInt = [](const float f) {
        uint32_t i;
        memcpy(&i, &f, sizeof(i));
        return i;
    };

    StageInfo stage;

    const auto isInput = [&](const CustomKernel::BindingParameter& binding) {
        return binding.type == CustomParamType::Input || binding.type == CustomParamType::InputBuffer ||
               binding.type == CustomParamType::Data;
    };

    const auto inputCount = std::count_if(begin(bindings), end(bindings), isInput);

    for (const auto& binding : bindings) {
        switch (binding.type) {
        case CustomParamType::InputBuffer: {
            const auto bufferIt = _buffers.find(binding.portIndex);
            VPU_THROW_UNLESS(bufferIt != _buffers.end(),
                             "Unable to deduce parameter '%s' for '%s' layer. "
                             "There is no output_buffer with port-index=%d defined.",
                             binding.argName, _node->description(), binding.portIndex);

            stage.inputs.push_back(bufferIt->second);
            stage.arguments.push_back(stage.inputs.size() - 1);
            break;
        }
        case CustomParamType::OutputBuffer: {
            VPU_THROW_UNLESS(_buffers.find(binding.portIndex) == _buffers.end(),
                             "Unable to deduce parameter '%s' for '%s' layer. "
                             "Can't add output_buffer with port-index=%d. "
                             "Buffer with that index already exists.",
                             binding.argName, _node->description(), binding.portIndex);

            const int bufferSize = parseBufferSize(binding);
            stage.outputs.emplace_back(true, bufferSize, binding.portIndex, binding.argName);
            stage.arguments.push_back(stage.outputs.size() - 1 + inputCount);
            break;
        }
        case CustomParamType::Data:
        case CustomParamType::Input: {
            VPU_THROW_UNLESS((uint32_t)binding.portIndex < _layerInputs.size(),
                             "Unable to deduce parameter '%s' for '%s' layer. "
                             "Can't find layer input with port-index=%d.",
                             binding.argName, _node->description(), binding.portIndex);

            stage.inputs.push_back(_layerInputs.at(binding.portIndex));
            stage.arguments.push_back(stage.inputs.size() - 1);
            break;
        }
        case CustomParamType::Output: {
            stage.outputs.emplace_back(false, 0, binding.portIndex, binding.argName);
            stage.arguments.push_back(stage.outputs.size() - 1 + inputCount);
            break;
        }
        case CustomParamType::Int:
        case CustomParamType::Float: {
            const auto cnnParam = _layerParam.find(binding.irSource);
            if (cnnParam != _layerParam.end()) {
                // parse cnnLayer param
                const auto param = [&]() -> std::string {
                    if (binding.portIndex < 0) {
                        VPU_THROW_UNLESS(parseNumber<float>(cnnParam->second).hasValue(),
                                         "Unable to deduce parameter '%s' for '%s' layer. "
                                         "Without "
                                         "port-index set, only viable "
                                         "size value is a whole integer number.",
                                         binding.argName, _node->description());
                        return cnnParam->second;
                    }

                    VPU_THROW_UNLESS(cnnParam->second.find(',') != std::string::npos,
                                     "Error while parsing CNNetwork parameter '%s' for '%s' "
                                     "layer: "
                                     "port-index=%d is set, "
                                     "but parameter is neither a tensor, nor an array type.",
                                     cnnParam->first, _node->description(), binding.portIndex);

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
                                     binding.argName, _node->description(), _node->get_friendly_name(),
                                     binding.irSource);
                    stage.arguments.push_back(val.get());
                } else {
                    const auto val = parseNumber<float>(param);
                    VPU_THROW_UNLESS(val.hasValue(),
                                     "Unable to deduce parameter '%s' for '%s' layer. "
                                     "Name is: '%s', parameter is: '%s'",
                                     binding.argName, _node->description(), _node->get_friendly_name(),
                                     binding.irSource);
                    stage.arguments.push_back(floatAsInt(val.get()));
                }
                // if not cnnLayer param, check if it is 'I.X' format param
            } else if (binding.irSource[1] == '.' && (binding.irSource[0] == 'I' || binding.irSource[0] == 'O')) {
                VPU_THROW_UNLESS(binding.irSource.length() == 3,
                                 "Unable to deduce parameter '%s' for '%s' layer."
                                 "Wrong source format",
                                 binding.argName, _node->description());

                const auto origDims = [&] {
                    if (binding.irSource[0] == 'I') {
                        return _node->get_input_shape(binding.portIndex);
                    }
                    return _node->get_output_shape(binding.portIndex);
                }();

                const auto dimLetter = toupper(binding.irSource[2]);
                auto dims = origDims;
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
                                 "Failed to parse source dimension from provided string "
                                 "'%s'",
                                 binding.argName, _node->description(), binding.irSource);

                auto dimValue = dims.at(dimPosition);
                stage.arguments.push_back(static_cast<uint32_t>(dimValue));
            } else {
                VPU_THROW_UNLESS(binding.portIndex < 0,
                                 "Unable to deduce parameter '%s' for '%s' layer: "
                                 "port-index=%d is set, "
                                 "but parameter is neither a tensor, nor an array type.",
                                 binding.argName, _node->description(), binding.portIndex);

                uint32_t number = 0;
                if (binding.type == CustomParamType::Int) {
                    const auto val = parseNumber<int>(binding.irSource);

                    VPU_THROW_UNLESS(val.hasValue(),
                                     "Unable to deduce parameter '%s' for '%s' layer. "
                                     "Name is: '%s', parameter is: '%s'",
                                     binding.argName, _node->description(), _node->get_friendly_name(),
                                     binding.irSource);

                    number = val.get();
                } else {
                    const auto val = parseNumber<float>(binding.irSource);

                    VPU_THROW_UNLESS(val.hasValue(),
                                     "Unable to deduce parameter '%s' for '%s' layer. "
                                     "Name is: '%s', parameter is: '%s'",
                                     binding.argName, _node->description(), _node->get_friendly_name(),
                                     binding.irSource);

                    number = floatAsInt(val.get());
                }

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

uint32_t CustomLayerParserNGraph::parseBufferSize(const CustomKernel::BindingParameter& binding) {
    const auto& source = binding.dimSource == CustomDimSource::Input ? _inputDescs : _outputDescs;
    const auto& desc = source[binding.dimIdx];
    const auto sizes = calcSizesFromParams(desc, {binding.bufferSizeRule}, _layerParam);
    return sizes[0];
}

void CustomLayerParserNGraph::addBuffer(int port, const mv::Data::TensorIterator& bufferIt) {
    _buffers.emplace(port, bufferIt);
}

}  // namespace vpu
