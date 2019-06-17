//
// Copyright 2016-2019 Intel Corporation.
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

#include <frontend_mcm.hpp>

#include <vector>
#include <memory>
#include <set>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <limits>

#include <ie_layers_internal.hpp>
#include <vpu/compile_env.hpp>
#include "ie_blob.h"

#include <precision_utils.h>

#ifdef ENABLE_MCM_COMPILER
#include "include/mcm/tensor/quantization_params.hpp"

namespace vpu {

namespace KmbPlugin {

namespace {

static std::unordered_map<int, char> DIM_NAMES({
    {3, 'W'},
    {2, 'H'},
    {1, 'C'},
    {0, 'N'}
});

}  // namespace

template<typename ResultType>
std::vector<ResultType> packBlobToVector(
        ie::Blob::Ptr blobPtr,
        int expectedSize) {
    IE_ASSERT(blobPtr != nullptr);

    std::vector<ResultType> blobData(expectedSize, 0);

    IE_ASSERT(expectedSize == blobPtr->size());

    ie::Precision blobPrecision = blobPtr->getTensorDesc().getPrecision();

    // TODO: add proper layout handling. for now, weights are assumed to have OIYX
    if (blobPrecision == ie::Precision::FP16) {
        const fp16_t* blobDataFP16 = blobPtr->cbuffer().as<const fp16_t*>();
        IE_ASSERT(blobDataFP16 != nullptr);

        for (int pos = 0; pos < expectedSize; pos++) {
            ResultType val = ie::PrecisionUtils::f16tof32(blobDataFP16[pos]);
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::FP32) {
        const float* blobDataFP32 = blobPtr->cbuffer().as<const float*>();
        IE_ASSERT(blobDataFP32 != nullptr);

        for (int pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataFP32[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::I8) {
        const int8_t* blobDataI8 = blobPtr->cbuffer().as<const int8_t*>();
        IE_ASSERT(blobDataI8 != nullptr);

        for (int pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataI8[pos];
            blobData[pos] = val;
        }
    } else {
        THROW_IE_EXCEPTION << "precision '" << blobPrecision << "' is not supported";
    }

    return blobData;
}

void FrontEndMcm::parseArgMax(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing ArgMax layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ArgMax layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseConvolution(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    bool is_quantized = false;
    //
    // Extract parameters
    //

    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    int kernelSizeX = convLayer->_kernel_x;
    int kernelSizeY = convLayer->_kernel_y;

    int kernelStrideX = convLayer->_stride_x;
    int kernelStrideY = convLayer->_stride_y;

    auto paddings = getPaddings(*convLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    int dilationX = convLayer->_dilation_x;
    int dilationY = convLayer->_dilation_y;
    if (dilationX != dilationY) {
        VPU_THROW_EXCEPTION << "kmb Convolution supports only eqiual dilationX and dilationY";
    }

    int groupSize = convLayer->_group;

    if (layer->blobs["weights"]->getTensorDesc().getPrecision() == InferenceEngine::Precision::I8 &&
        layer->blobs["w-scale"] != nullptr) {
        is_quantized = true;
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    DataDesc outDesc(layerOutput->getTensorDesc());

    mv::Data::TensorIterator mvConv;
    // input should be already quantized
    const mv::QuantizationParams inputQuantParams = {{}, {}, {}, {}};

    mv::Data::TensorIterator mvWeights;

    env.log->debug("Convolution orig: '%s' from '%s' ", convLayer->name, input->getMcmNode()->getName());
    if (groupSize > 1 && groupSize == input->desc().dim(Dim::C) && groupSize != outDesc.dim(Dim::C)) {
        auto weightsShape = {static_cast<std::size_t>(kernelSizeX),
                             static_cast<std::size_t>(kernelSizeY),
                             static_cast<std::size_t>(input->desc().dim(Dim::C)),
                             1lu};
        if (is_quantized) {
            // TODO: create per layer test
            int weightsSize = kernelSizeX * kernelSizeY * input->desc().dim(Dim::C);
            auto weightsData = packBlobToVector<int64_t>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constantInt(weightsData,
                                              weightsShape,
                                              mv::DType("Int8"),
                                              mv::Order(mv::Order::getColMajorID(4)));
            // extract and set quantization params
            auto wscale = packBlobToVector<double>(layer->blobs["w-scale"], layer->blobs["w-scale"]->size());
            mv::QuantizationParams weightsQuantParams(
                    mv::utils::generateSequence<int64_t>(wscale.size(), 0, 0),
                    wscale,
                    mv::utils::generateSequence<double>(wscale.size(), 0, 0),
                    mv::utils::generateSequence<double>(wscale.size(), 1, 0));
            mvWeights->set<mv::QuantizationParams>("quantizationParams", weightsQuantParams);
        } else {
            int weightsSize = kernelSizeX * kernelSizeY * input->desc().dim(Dim::C);
            auto weightsData = packBlobToVector<double>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constant(weightsData,
                                           weightsShape,
                                           mv::DType("Float16"),
                                           mv::Order(mv::Order::getColMajorID(4)));
        }

        mvConv = _modelMcm.depthwiseConv(input->getMcmNode(),
                                         mvWeights,
                                        {static_cast<uint16_t>(kernelStrideX),
                                         static_cast<uint16_t>(kernelStrideY)},
                                        {static_cast<uint16_t>(padLeft),
                                         static_cast<uint16_t>(padRight),
                                         static_cast<uint16_t>(padTop),
                                         static_cast<uint16_t>(padBottom)},
                                         static_cast<unsigned>(dilationX),
                                         inputQuantParams,
                                         convLayer->name);
    } else {
        auto weightsShape = {static_cast<std::size_t>(kernelSizeX),
                             static_cast<std::size_t>(kernelSizeY),
                             static_cast<std::size_t>(input->desc().dim(Dim::C)),
                             static_cast<std::size_t>(outDesc.dim(Dim::C)) / groupSize};

        if (is_quantized) {
            int weightsSize = kernelSizeX * kernelSizeY * input->desc().dim(Dim::C) * outDesc.dim(Dim::C) / groupSize;
            auto weightsData = packBlobToVector<int64_t>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constantInt(weightsData,
                                              weightsShape,
                                              mv::DType("Int8"),
                                              mv::Order("NCWH"));
            // extract and set quantization params
            auto wscale = packBlobToVector<double>(layer->blobs["w-scale"], layer->blobs["w-scale"]->size());
            mv::QuantizationParams weightsQuantParams(
                    mv::utils::generateSequence<int64_t>(wscale.size(), 0, 0),
                    wscale,
                    mv::utils::generateSequence<double>(wscale.size(), 0, 0),
                    mv::utils::generateSequence<double>(wscale.size(), 1, 0));
            mvWeights->set<mv::QuantizationParams>("quantizationParams", weightsQuantParams);
        } else {
            int weightsSize = kernelSizeX * kernelSizeY * input->desc().dim(Dim::C) * outDesc.dim(Dim::C) / groupSize;
            auto weightsData = packBlobToVector<double>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constant(weightsData,
                                           weightsShape,
                                           mv::DType("Float16"),
                                           mv::Order("NCWH"));
        }

        mvConv = _modelMcm.conv(input->getMcmNode(),
                                mvWeights,
                               {static_cast<uint16_t>(kernelStrideX),
                                static_cast<uint16_t>(kernelStrideY)},
                               {static_cast<uint16_t>(padLeft),
                                static_cast<uint16_t>(padRight),
                                static_cast<uint16_t>(padTop),
                                static_cast<uint16_t>(padBottom)},
                                static_cast<unsigned>(dilationX),
                                static_cast<unsigned>(groupSize),
                                inputQuantParams,
                                convLayer->name);
    }

    auto conv = std::make_shared<McmNodeObject>(mvConv, outDesc);

    _nodes.push_back(conv);
    bindData(conv, layerOutput);

    env.log->debug("parsed to mcmModel as '%s'", mvConv->getName());
}

void FrontEndMcm::parsePooling(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

    int kernelSizeX = poolLayer->_kernel_x;
    int kernelSizeY = poolLayer->_kernel_y;

    int kernelStrideX = poolLayer->_stride_x;
    int kernelStrideY = poolLayer->_stride_y;

    auto paddings = getPaddings(*poolLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    auto poolType = poolLayer->_type;

    env.log->debug("Pooling orig: '%s' from '%s'", poolLayer->name, inputs[0]->getMcmNode()->getName());
    mv::Data::TensorIterator mvPooling;
    if (poolType == ie::PoolingLayer::AVG) {
        mvPooling = _modelMcm.averagePool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX),
             static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX),
             static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft),
             static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop),
             static_cast<uint16_t>(padBottom)},
            true, "", "floor", {{}, {}, {}, {}},
            poolLayer->name);
    } else {
        mvPooling = _modelMcm.maxPool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX),
             static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX),
             static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft),
             static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop),
             static_cast<uint16_t>(padBottom)},
            true, "", "floor", {{}, {}, {}, {}},
            poolLayer->name);
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto pool = std::make_shared<McmNodeObject>(mvPooling, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(pool);
    bindData(pool, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvPooling->getName());
}

void FrontEndMcm::parseFullyConnected(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& _layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::FullyConnectedLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];

    //
    // Create const datas
    //
    int weightsSize = input->desc().dim(Dim::W, 1) *
                      input->desc().dim(Dim::H, 1) *
                      input->desc().dim(Dim::C) *
                      static_cast<int>(layer->_out_num);
    std::vector<double> weightsData = packBlobToVector<double>(layer->blobs["weights"], weightsSize);

    auto mvWeights = _modelMcm.constant(weightsData,
            {inputs[0]->getMcmNode()->getShape().totalSize(), static_cast<std::size_t>(layer->_out_num)},
            mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)));

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    env.log->debug("FullyConnected orig: '%s' from '%s'",
            layer->name, input->getMcmNode()->getName());
    auto mvFullyConnected = _modelMcm.fullyConnected(input->getMcmNode(), mvWeights, {{}, {}, {}, {}}, layer->name);
    auto fullyConnected = std::make_shared<McmNodeObject>(mvFullyConnected, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(fullyConnected);
    bindData(fullyConnected, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvFullyConnected->getName());
}

namespace {

union FloatUint {
    float flt;
    uint32_t unt;
};

}  // namespace

void FrontEndMcm::parseReLU(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto reluLayer = std::dynamic_pointer_cast<ie::ReLULayer>(layer);
    IE_ASSERT(reluLayer != nullptr);

    env.log->debug("ReLU orig: '%s' from '%s'", reluLayer->name, inputs[0]->getMcmNode()->getName());

    float negativeSlope = reluLayer->negative_slope;
    mv::Data::TensorIterator mvRelu;
    if (std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon()) {
        mvRelu = _modelMcm.relu(inputs[0]->getMcmNode(), {{}, {}, {}, {}}, reluLayer->name);
    } else {
        // TODO FIXME: unsigned int alpha should be fixed or clarified
        mvRelu = _modelMcm.leakyRelu(inputs[0]->getMcmNode(),
                negativeSlope,
                reluLayer->name);
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto relu = std::make_shared<McmNodeObject>(mvRelu, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(relu);
    bindData(relu, layerOutput);
    env.log->debug("parsed to mcmModel as '%s", mvRelu->getName());
}

void FrontEndMcm::parseSoftMax(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& _layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::SoftMaxLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(layer->axis < inputs[0]->desc().numDims());

    env.log->debug("Softmax orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[layer->axis];
    auto mvSoftmax = _modelMcm.softmax(inputs[0]->getMcmNode(), mcmAxis, {{}, {}, {}, {}}, layer->name);

    auto layerOutput = _layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto softmax = std::make_shared<McmNodeObject>(mvSoftmax, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(softmax);
    bindData(softmax, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvSoftmax->getName());
}

void FrontEndMcm::parseGRN(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing GRN layer %s", layer->name);
    VPU_THROW_EXCEPTION << "GRN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMVN(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing MVN layer %s", layer->name);
    VPU_THROW_EXCEPTION << "MVN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseNorm(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto normLayer = std::dynamic_pointer_cast<ie::NormLayer>(layer);
    IE_ASSERT(normLayer != nullptr);

    env.log->debug("LRN orig: '%s' from '%s'", normLayer->name, inputs[0]->getMcmNode()->getName());
    auto mvLRN = _modelMcm.localResponseNormalization(inputs[0]->getMcmNode(),
            normLayer->_size, normLayer->_k, normLayer->name);

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto LRN = std::make_shared<McmNodeObject>(mvLRN, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(LRN);
    bindData(LRN, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvLRN->getName());

    // TODO: add parsing following parameters
    // stage->attrs().set<float>("alpha", layer->_alpha);
    // stage->attrs().set<float>("beta", layer->_beta);
}

void FrontEndMcm::parsePower(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Power layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Power layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseScale(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& _layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(_layer);
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(layer->_weights != nullptr);

    if (layer->_broadcast != 0) {
        VPU_THROW_EXCEPTION <<
            "Layer " << layer->name << " doesn't support broadcast param";
    }

    auto input = inputs[0];

    int weightsSize = input->desc().dim(Dim::C);
    auto weightsData = packBlobToVector<double>(layer->_weights, weightsSize);

    mv::Shape weightsShape = { static_cast<std::size_t>(input->desc().dim(Dim::C))};
    auto mvWeights = _modelMcm.constant(
            weightsData,
            weightsShape,
            mv::DType("Float16"), mv::Order("W"));

    env.log->debug("ScaleShift orig: '%s' from '%s' ", layer->name, input->getMcmNode()->getName());
    auto mvScale = _modelMcm.scale(input->getMcmNode(), mvWeights, {{}, {}, {}, {}}, layer->name);
    auto mvScaleShift = mvScale;

    env.log->debug("ScaleShift orig: '%s' Scale part (%s) added to mcmModel", layer->name, mvScaleShift->getName());

    if (layer->_biases != nullptr) {
        int biasesSize = input->desc().dim(Dim::C);
        auto biasData = packBlobToVector<double>(layer->_biases, biasesSize);

        auto mvBias = _modelMcm.constant(
                weightsData,
                weightsShape,
                mv::DType("Float16"), mv::Order("W"));
        mvScaleShift = _modelMcm.bias(mvScale, mvBias, {{}, {}, {}, {}}, layer->name + ":bias");
        env.log->debug("ScaleShift orig: '%s' Bias part (%s) added to mcmModel", layer->name, mvScaleShift->getName());
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto scaleShift = std::make_shared<McmNodeObject>(mvScaleShift, DataDesc(layerOutput->getTensorDesc()));
    _nodes.push_back(scaleShift);
    bindData(scaleShift, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvScaleShift->getName());
}

void FrontEndMcm::parsePermute(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    IE_ASSERT(inputs.size() == 1);

    auto ieOrder = layer->GetParamAsInts("order");

    std::string newOrder;

//  4d NCHW inputs are supported
    for (size_t i = 0; i < ieOrder.size(); i++) {
        newOrder += DIM_NAMES[ieOrder[ieOrder.size() - 1 - i]];
    }

    env.log->debug("Permute orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    auto mvPerm = _modelMcm.permute(inputs[0]->getMcmNode(), mv::Order(newOrder), layer->name);

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto perm = std::make_shared<McmNodeObject>(mvPerm, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(perm);
    bindData(perm, layerOutput);
    env.log->debug("parsed to mcmModel as '%s", mvPerm->getName());
}

void FrontEndMcm::parseDetectionOutput(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing DetectionOutput layer %s", layer->name);
    VPU_THROW_EXCEPTION << "DetectionOutput layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseEltwise(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    if (inputs.size() != 2) {
        VPU_THROW_EXCEPTION << "Eltwise with 2 inputs is only supported by kmbPlugin";
    }

    auto eltwiseLayer = std::dynamic_pointer_cast<ie::EltwiseLayer>(layer);
    IE_ASSERT(eltwiseLayer != nullptr);

    auto stageType = StageType::None;
    auto addCoefficient0 = 1.0f;
    auto addCoefficient1 = 1.0f;
    switch (eltwiseLayer->_operation) {
    case ie::EltwiseLayer::eOperation::Sum:
    case ie::EltwiseLayer::eOperation::Sub:
        addCoefficient1 = -1.0f;
        if (eltwiseLayer->coeff.size() > 0) {
            addCoefficient0 *= eltwiseLayer->coeff[0];
        }
        if (eltwiseLayer->coeff.size() > 1) {
            addCoefficient1 *= eltwiseLayer->coeff[1];
        }
        if (std::abs(addCoefficient0) != 1.0f || std::abs(addCoefficient1) != 1.0f ||
                (addCoefficient0 == -1.0f && addCoefficient1 == -1.0f)) {
            VPU_THROW_EXCEPTION << eltwiseLayer->name <<
                    " Eltwise Sum/Sub operations with such coefficients is not supported by kmbPlugin";
        }
        stageType = StageType::Sum;
        break;
    case ie::EltwiseLayer::eOperation::Prod:
        stageType = StageType::Prod;
        break;
    default:
        VPU_THROW_EXCEPTION << "Eltwise operation" << eltwiseLayer->_operation << " is not supported";
    }

    if (stageType != StageType::Sum && !eltwiseLayer->coeff.empty()) {
        VPU_THROW_EXCEPTION << layer->name << " coefficients (1 and -1) are only supported for Sum/Sub operations.";
    }

    env.log->debug("eltwise orig: '%s' from '%s' and '%s'",
            eltwiseLayer->name, inputs[0]->getMcmNode()->getName(), inputs[1]->getMcmNode()->getName());
    mv::Data::TensorIterator mvEltwise;
    if (addCoefficient0 == 1.0f && addCoefficient1 == 1.0f) {
        mvEltwise = _modelMcm.add(inputs[0]->getMcmNode(), inputs[1]->getMcmNode(), {{}, {}, {}, {}}, eltwiseLayer->name);
    } else {
        if (addCoefficient0 == 1.0f && addCoefficient1 == -1.0f) {
            mvEltwise = _modelMcm.subtract(inputs[0]->getMcmNode(), inputs[1]->getMcmNode(), {{}, {}, {}, {}}, eltwiseLayer->name);
        } else {
            mvEltwise = _modelMcm.subtract(inputs[1]->getMcmNode(), inputs[0]->getMcmNode(), {{}, {}, {}, {}}, eltwiseLayer->name);
        }
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto eltwise = std::make_shared<McmNodeObject>(mvEltwise, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(eltwise);
    bindData(eltwise, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvEltwise->getName());
}

void FrontEndMcm::parseSigmoid(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Sigmoid layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Sigmoid layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTanH(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing TanH layer %s", layer->name);
    VPU_THROW_EXCEPTION << "TanH layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePReLU(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing PReLU layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseBatchNorm(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing PReLU layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDeconvolution(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Deconvolution( layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Deconvolution layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCopy(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Copy layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Copy layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseELU(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing ELU layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ELU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCrop(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Crop layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Crop layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTile(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Tile layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Tile layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseNormalize(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Normalize layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Normalize layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseRegionYolo(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing RegionYolo layer %s", layer->name);
    VPU_THROW_EXCEPTION << "RegionYolo layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseReorgYolo(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing ReorgYolo layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ReorgYolo layer is not supported by kmbPlugin";
}








void FrontEndMcm::parseBias(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
//    const auto& env = CompileEnv::get();
//    env.log->debug("Parsing Bias layer %s", layer->name);
//    VPU_THROW_EXCEPTION << "Bias layer is not supported by kmbPlugin";

    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 2);

    auto biasLayer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(layer);
//    IE_ASSERT(layer != nullptr);
//    IE_ASSERT(layer->_weights != nullptr);
//
//    if (layer->_broadcast != 0) {
//        VPU_THROW_EXCEPTION <<
//            "Layer " << layer->name << " doesn't support broadcast param";
//    }

    auto input = inputs[0];
//    auto biases = inputs[1];
//    auto output = outputs[0];

//    auto input = inputs[0];
//
//    int weightsSize = input->desc().dim(Dim::C);
//    auto weightsData = packBlobToVector<double>(layer->_weights, weightsSize);
//
//    mv::Shape weightsShape = { static_cast<std::size_t>(input->desc().dim(Dim::C))};
//    auto mvWeights = _modelMcm.constant(
//            weightsData,
//            weightsShape,
//            mv::DType("Float16"), mv::Order("W"));

    env.log->debug("ScaleShift(Bias) orig: '%s' from '%s' ", biasLayer->name, input->getMcmNode()->getName());
//    auto mvScale = _modelMcm.scale(input->getMcmNode(), mvWeights, {{}, {}, {}, {}}, layer->name);
//    auto mvScaleShift = mvScale;
//
//    env.log->debug("ScaleShift orig: '%s' Scale part (%s) added to mcmModel", layer->name, mvScaleShift->getName());
//
//    if (layer->_biases != nullptr) {
        mv::Shape biasShape = { static_cast<std::size_t>(input->desc().dim(Dim::C))};
        int biasesSize = input->desc().dim(Dim::C);
        auto biasData = packBlobToVector<double>(biasLayer->_biases, biasesSize);

        auto mvBiasValues = _modelMcm.constant(
                biasData,
                biasShape,
                mv::DType("Float16"), mv::Order("W"));
        auto mvBias = _modelMcm.bias(input->getMcmNode(), mvBiasValues, {{}, {}, {}, {}}, biasLayer->name + ":bias");
//        env.log->debug("ScaleShift orig: '%s' Bias part (%s) added to mcmModel", biasLayer->name, mvBias->getName());
//    }

    auto layerOutput = biasLayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto bias = std::make_shared<McmNodeObject>(mvBias, DataDesc(layerOutput->getTensorDesc()));
    _nodes.push_back(bias);
    bindData(bias, layerOutput);
    env.log->debug("parsed to mcmModel as '%s'", mvBias->getName());
}

void FrontEndMcm::parseCTCDecoder(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing CTCDecoder layer %s", layer->name);
    VPU_THROW_EXCEPTION << "CTCDecoder layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseInterp(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Interp layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Interp layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseClamp(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& _layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ClampLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    env.log->debug("Clamp orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    auto mvClamp = _modelMcm.clamp(inputs[0]->getMcmNode(), layer->min_value, layer->max_value, layer->name);

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto clamp = std::make_shared<McmNodeObject>(mvClamp, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(clamp);
    bindData(clamp, layerOutput);
    env.log->debug("parsed to mcmModel as '%s", mvClamp->getName());
}

void FrontEndMcm::parseProposal(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Proposal layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Proposal layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseROIPooling(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing ROIPooling layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePSROIPooling(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing PSROIPooling layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PSROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCustom(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Custom layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Custom layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMTCNN(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing MTCNN layer %s", layer->name);
    VPU_THROW_EXCEPTION << "MTCNN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePad(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Pad layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Pad layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseResample(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Resample layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Resample layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseLSTMCell(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing LSTMCell layer %s", layer->name);
    VPU_THROW_EXCEPTION << "LSTMCell layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBox(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing PriorBox layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PriorBox layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBoxClustered(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing PriorBoxClustered layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PriorBoxClustered layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseReshape(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    DataDesc vpuDesc(layerOutput->getTensorDesc());

    mv::Shape newShape = {static_cast<std::size_t>(vpuDesc.dim(Dim::W, 1)),
                            static_cast<std::size_t>(vpuDesc.dim(Dim::H, 1)),
                            static_cast<std::size_t>(vpuDesc.dim(Dim::C, 1)),
                            static_cast<std::size_t>(vpuDesc.dim(Dim::N, 1))};

    env.log->debug("Reshape orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    auto mvReshape = _modelMcm.reshape(inputs[0]->getMcmNode(), newShape, "", layer->name);

    auto reshape = std::make_shared<McmNodeObject>(mvReshape, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(reshape);
    bindData(reshape, layerOutput);
    env.log->debug("parsed to mcmModel as '%s", mvReshape->getName());
}

void FrontEndMcm::parseConcat(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& _layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    IE_ASSERT(!inputs.empty());

    auto layer = std::dynamic_pointer_cast<ie::ConcatLayer>(_layer);
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(layer->_axis < inputs[0]->desc().numDims());

    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[layer->_axis];
    std::vector<mv::Data::TensorIterator> concatInputs;

    for (size_t i = 0; i < inputs.size(); ++i) {
        concatInputs.push_back(inputs[i]->getMcmNode());
    }

    env.log->debug("Concat orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    auto mvConcat = _modelMcm.concat(concatInputs, mcmAxis, {{}, {}, {}, {}}, layer->name + ":step0");

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto concat = std::make_shared<McmNodeObject>(mvConcat, DataDesc(layerOutput->getTensorDesc()));

    _nodes.push_back(concat);
    bindData(concat, layerOutput);
    env.log->debug("parsed to mcmModel as '%s", mvConcat->getName());
}

void FrontEndMcm::parseSplit(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    const auto& env = CompileEnv::get();
    env.log->debug("Parsing Split layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Split layer is not supported by kmbPlugin";
}

void FrontEndMcm::checkNetwork(const ie::CNNNetwork& network) {
    const auto& env = CompileEnv::get();

    auto networkPrecision = network.getPrecision();
    if (networkPrecision != ie::Precision::FP16) {
        if (networkPrecision != ie::Precision::FP32 || !env.config.allowFP32Models) {
            VPU_THROW_EXCEPTION << "Unsupported network precision : " << networkPrecision;
        }
    }

    _ieNetworkParser.networkInputs = network.getInputsInfo();
    _ieNetworkParser.networkOutputs = network.getOutputsInfo();

    if (_ieNetworkParser.networkInputs.empty()) {
        VPU_THROW_EXCEPTION << "No inputs detected in network " << network.getName();
    }
    if (_ieNetworkParser.networkOutputs.empty()) {
        VPU_THROW_EXCEPTION << "No outputs detected in network " << network.getName();
    }

    for (const auto& netInput : _ieNetworkParser.networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);

        auto inputPrecision = inputInfo->getPrecision();

        if (inputPrecision != ie::Precision::U8 &&
            inputPrecision != ie::Precision::FP16 &&
            inputPrecision != ie::Precision::FP32) {
            THROW_IE_EXCEPTION << "[PARAMETER_MISMATCH] Unsupported input precision: " << inputPrecision.name() << "!";
        }
    }

    for (const auto& netOutput : _ieNetworkParser.networkOutputs) {
        auto outputData = netOutput.second;
        IE_ASSERT(outputData != nullptr);

        auto outputPrecision = outputData->getPrecision();

        if (outputPrecision != ie::Precision::FP16 &&
            outputPrecision != ie::Precision::FP32) {
            THROW_IE_EXCEPTION << "[PARAMETER_MISMATCH] Unsupported output precision: " << outputPrecision.name() << "!";
        }
    }
}

}  // namespace KmbPlugin

}  // namespace vpu
#endif
