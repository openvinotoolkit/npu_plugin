//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_set>
#include <utility>
#include <algorithm>

#include <vpu/custom_layer.hpp>
#include <vpu/utils/simple_math.hpp>

namespace vpu {

namespace {

class KernelBinaryContent final : public DataContent {
public:
    explicit KernelBinaryContent(const std::string& blob) : _blob(blob) {
        IE_ASSERT(!_blob.empty());
    }

    const void* getRaw() const override {
        IE_ASSERT(_desc.totalDimSize() * _desc.elemSize() == _blob.length());
        return _blob.data();
    }

private:
    std::string _blob;
};

class CustomStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CustomStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        const auto& inputOrders = attrs().get<std::map<int, DimsOrder>>("inputOrders");
        const auto& outputOrders = attrs().get<std::map<int, DimsOrder>>("outputOrders");

        for (const auto& inEdge : _inputEdges) {
            // last input is always OpenCL binary, so use it as is.
            if (inEdge->portInd() == _inputEdges.size() - 1) {
                break;
            }

            auto it = inputOrders.find(inEdge->portInd());
            if (it != inputOrders.end()) {
                auto requiredOrder = it->second;
                _orderInfo.setInput(inEdge, requiredOrder);
            }
        }

        for (const auto& outEdge : _outputEdges) {
            auto it = outputOrders.find(outEdge->portInd());
            if (it != outputOrders.end()) {
                auto requiredOrder = it->second;
                _orderInfo.setOutput(outEdge, requiredOrder);
            }
        }
    }

    void getDataStridesRequirementsImpl() const override {
        for (const auto& inEdge : _inputEdges) {
            // last input is always OpenCL binary, so use it as is.
            if (inEdge->portInd() == _inputEdges.size() - 1) {
                break;
            }

            _stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : _outputEdges) {
            _stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        for (const auto& inEdge : _inputEdges) {
            // last input is always OpenCL binary, so use it as is.
            if (inEdge->portInd() == _inputEdges.size() - 1) {
                break;
            }

            _batchInfo.setInput(inEdge, BatchSupport::Split);
        }
        for (const auto& outEdge : _outputEdges) {
            _batchInfo.setOutput(outEdge, BatchSupport::Split);
        }
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& customLayer = attrs().get<CustomLayer::Ptr>("customLayer");
        const auto& gws = attrs().get<SmallVector<int, 3>>("gws");
        const auto& lws = attrs().get<SmallVector<int, 3>>("lws");

        //
        // GWG, LWG, Offs
        //

        for (auto x : gws) {
            serializer.append(static_cast<uint32_t>(x));
        }

        for (auto x : lws) {
            serializer.append(static_cast<uint32_t>(x));
        }

        for (int i = 0; i < lws.size(); ++i) {
            serializer.append(static_cast<uint32_t>(0));
        }

        serializer.append(static_cast<uint32_t>(customLayer->maxShaves()));

        //
        // Entry point
        //
        IE_ASSERT(customLayer->stageNumInputs() >= 0);
        serializer.append(static_cast<uint32_t>(customLayer->stageNumInputs()));
        serializer.append(static_cast<uint32_t>(customLayer->kernelAddress(lws[0])));

        //
        // Total number of blobs
        //

        serializer.append(static_cast<int32_t>(_inputEdges.size() + _outputEdges.size()));

        //
        // Number of kernel parameters
        //

        serializer.append(static_cast<uint32_t>(customLayer->parameters().size()));

        //
        // Parameters & relocation info
        //

        std::map<std::string, CustomLayer::KernelParam> b2b;
        for (const auto& kp : customLayer->bindings()) {
            b2b[kp.argName] = kp;
        }

        IE_ASSERT(_origLayer != nullptr);

        int tensorId = 0;
        for (const auto& kp : customLayer->parameters()) {
            const auto& parameter = b2b[kp];

            switch (parameter.type) {
                case CustomParamType::Input:
                case CustomParamType::Output:
                case CustomParamType::InputBuffer:
                case CustomParamType::OutputBuffer:
                {
                    serializer.append(static_cast<uint32_t>(0));
                    serializer.append(static_cast<uint32_t>(tensorId));
                    tensorId = (tensorId+1 == customLayer->stageNumInputs() ? tensorId+2 : tensorId+1);
                    break;
                }
                case CustomParamType::Data:
                {
                    // TODO: handle data
                    break;
                }
                case CustomParamType::Int:
                case CustomParamType::Float:
                {
                    if (_origLayer->params.find(parameter.irSource) != _origLayer->params.end()) {
                        if (parameter.type == CustomParamType::Int) {
                            serializer.append(static_cast<int32_t>(std::stoi(_origLayer->params[parameter.irSource]) ));
                            serializer.append(static_cast<int32_t>(-1));
                        } else {
                            serializer.append(static_cast<float>(std::stof(_origLayer->params[parameter.irSource]) ));
                            serializer.append(static_cast<int32_t>(-2));
                        }
                        break;
                    } else {
                        auto pos = parameter.irSource.find_first_of('.');
                        if (pos != std::string::npos) {
                            auto blob = parameter.irSource.substr(0, pos);
                            auto dim = parameter.irSource.substr(pos + 1, std::string::npos);

                            ie::DataPtr origData;
                            if (blob == "I") {
                                origData = _origLayer->insData[0].lock();
                            } else {
                                origData = _origLayer->outData[0];
                            }
                            IE_ASSERT(origData != nullptr);

                            auto dims = origData->getDims();

                            const std::map<char, int> vars = {
                                { 'b', 0 }, { 'B', 0 },
                                { 'f', 1 }, { 'F', 1 },
                                { 'y', 2 }, { 'Y', 2 },
                                { 'x', 3 }, { 'X', 3 },
                            };

                            if (vars.find(dim[0]) != vars.end()) {
                                auto res = dims.at(vars.at(dim[0]));

                                serializer.append(static_cast<uint32_t>(res));
                                serializer.append(static_cast<int32_t>(-1));
                            } else {
                                VPU_THROW_EXCEPTION
                                    << "Unable to deduce parameter " << parameter.argName << " for "
                                    << _origLayer->type <<" layer. Name is: " << _origLayer->name;
                            }

                            break;
                        }

                        VPU_THROW_EXCEPTION
                            << "Unable to deduce parameter " << parameter.argName << " for "
                            << _origLayer->type <<" layer. Name is: " << _origLayer->name;
                    }
                }
                default:
                    VPU_THROW_EXCEPTION
                        << "Unable to deduce parameter " << parameter.argName << " for "
                        << _origLayer->type <<" layer. Name is: " << _origLayer->name;
            }
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_tempBufferEdges.empty());

        for (const auto& inEdge : _inputEdges) {
            inEdge->input()->serializeOldBuffer(handle_from_this(), serializer);
        }

        for (const auto& outEdge : _outputEdges) {
            outEdge->output()->serializeOldBuffer(handle_from_this(), serializer);
        }
    }
};

}  // namespace

static void calcSizesFromParams(const DataDesc &desc, const SmallVector<std::string> &bufferSizeRules, SmallVector<int, 3> &sizes) {
    // assume output tensor is dimension source by default
    auto batchDim = desc.dim(Dim::N, 1);
    auto featureDim = desc.dim(Dim::C, 1);
    auto yDim = desc.dim(Dim::H, 1);
    auto xDim = desc.dim(Dim::W, 1);

    const std::map<char, int> vars = {
        { 'b', batchDim },   { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };

    sizes.reserve(std::max<size_t>(bufferSizeRules.size(), 3));
    for (const auto& rule : bufferSizeRules) {
        SimpleMathExpression expr;
        expr.setVariables(vars);
        expr.parse(rule);
        sizes.emplace_back(expr.evaluate());
    }
    while (sizes.size() < 3) {
        sizes.emplace_back(1);
    }
}

void FrontEnd::parseCustom(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(outputs.size() == 1);

    std::vector<CustomLayer::Ptr> customLayersForType;

    if (_customLayers.count(layer->type) > 0) {
        customLayersForType.push_back((_customLayers.find(layer->type))->second);
    } else if (_customLayers.count(layer->type + "@stage_0") > 0) {
        int stageNum = 0;
        while (_customLayers.count(layer->type + "@stage_" + std::to_string(stageNum)) > 0) {
            customLayersForType.push_back((_customLayers.find(layer->type + "@stage_" + std::to_string(stageNum)))->second);
            stageNum++;
        }
    } else {
        IE_ASSERT(false);
    }

    DataVector tempBuffs;
    for (size_t bufferPortIndex = 0; bufferPortIndex == tempBuffs.size(); bufferPortIndex++) {
        for (size_t stageNum = 0; stageNum < customLayersForType.size(); stageNum++) {
            for (auto& param : customLayersForType[stageNum]->bindings()) {
                if (param.portIndex == bufferPortIndex &&
                   (param.type == CustomParamType::InputBuffer || param.type == CustomParamType::OutputBuffer)) {
                    SmallVector<int, 3> sizes;
                    auto desc = (param.dimSource == CustomDimSource::Input) ? inputs[param.dimIdx]->desc() : outputs[param.dimIdx]->desc();
                    calcSizesFromParams(desc, param.bufferSizeRules, sizes);
                    auto mvn_buf = model->addNewData("custom_mvn_buf", DataDesc({sizes[0], sizes[1], sizes[2], 1}));
                    tempBuffs.emplace_back(mvn_buf);
                }
            }
        }
    }

    for (int stage_num = 0; stage_num < customLayersForType.size(); stage_num++) {
        auto customLayer = customLayersForType[stage_num];

        auto kernelBinaryDesc = DataDesc({customLayer->kernelBinary().length()});
        kernelBinaryDesc.setType(DataType::U8);

        auto kernelBinary = model->addConstData(
            layer->name + "@kernelBinary",
            kernelBinaryDesc,
            std::make_shared<KernelBinaryContent>(customLayer->kernelBinary()));

        DataVector stageInputs;
        for (int inputPortInex = 0; inputPortInex == stageInputs.size(); inputPortInex++) {
            for (auto& param : customLayer->bindings()) {
                if (param.portIndex == inputPortInex) {
                    if (param.type == CustomParamType::Input) {
                        stageInputs.emplace_back(inputs[inputPortInex]);
                    } else if (param.type == CustomParamType::InputBuffer) {
                        stageInputs.emplace_back(tempBuffs[inputPortInex]);
                    }
                }
            }
        }
        customLayer->setStageNumInputs(stageInputs.size());
        stageInputs.emplace_back(std::move(kernelBinary));

        DataVector stageOutputs;
        for (int outputPortIndex = 0; outputPortIndex == stageOutputs.size(); outputPortIndex++) {
            for (auto& param : customLayer->bindings()) {
                if (param.portIndex == outputPortIndex) {
                    if (param.type == CustomParamType::Output) {
                        stageOutputs.emplace_back(outputs[outputPortIndex]);
                    } else if (param.type == CustomParamType::OutputBuffer) {
                        stageOutputs.emplace_back(tempBuffs[outputPortIndex]);
                    }
                }
            }
        }

        auto stage = model->addNewStage<CustomStage>(
            layer->name + ((customLayersForType.size() == 1) ? "" : "@stage_" + std::to_string(stage_num)),
            StageType::Custom,
            layer,
            stageInputs,
            stageOutputs);

        stage->attrs().set("customLayer", customLayer);

        SmallVector<int, 3> gws;
        SmallVector<int, 3> lws;
        auto dimSource = (customLayer->dimSource() == CustomDimSource::Input) ? inputs : outputs;
        calcSizesFromParams(dimSource[customLayer->dimSourceIndex()]->desc(), customLayer->globalSizeRules(), gws);
        calcSizesFromParams(dimSource[customLayer->dimSourceIndex()]->desc(), customLayer->localSizeRules(), lws);

        stage->attrs().set("gws", gws);
        stage->attrs().set("lws", lws);

        std::map<int, DimsOrder> inputOrders;
        std::map<int, DimsOrder> outputOrders;

        std::map<std::string, CustomLayer::KernelParam> b2b;
        for (const auto& kp : customLayer->bindings()) {
            b2b[kp.argName] = kp;
        }

        const std::map<CustomDataFormat, DimsOrder> formats = {
            { CustomDataFormat::BYXF, DimsOrder::NHWC },
            { CustomDataFormat::BFYX, DimsOrder::NCHW }
        };

        for (const auto& kp : customLayer->parameters()) {
            const auto& parameter = b2b[kp];

            if (parameter.type == CustomParamType::Input) {
                auto it = formats.find(parameter.format);
                if (it != formats.end()) {
                    auto requiredOrder = it->second;
                    inputOrders[parameter.portIndex] = requiredOrder;
                }
            }

            if (parameter.type == CustomParamType::Output) {
                auto it = formats.find(parameter.format);
                if (it != formats.end()) {
                    auto requiredOrder = it->second;
                    outputOrders[parameter.portIndex] = requiredOrder;
                }
            }
        }

        stage->attrs().set("inputOrders", std::move(inputOrders));
        stage->attrs().set("outputOrders", std::move(outputOrders));
    }
}

}  // namespace vpu
