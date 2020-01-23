//
// Copyright 2019-2020 Intel Corporation.
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

#include "quantization_helpers.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#ifdef ENABLE_MCM_COMPILER
#include "include/mcm/tensor/quantization_params.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace vpu {

namespace QuantizationHelpers {

double inf = std::numeric_limits<double>::infinity();
mv::QuantizationParams initialQuantParams = {{0}, {1}, {-inf}, {inf}};

static double clamp(const double& v, const double& lo, const double& hi) { return (v < lo) ? lo : (hi < v) ? hi : v; }

bool isPostOp(const InferenceEngine::CNNLayerPtr& layer) {
    return ((layer->type == "ReLU") || (layer->type == "Clamp"));
}

std::vector<float> getBlobValue(const InferenceEngine::CNNLayerPtr& constantLayer) {
    const auto blob = constantLayer->blobs.begin()->second;
    auto buffer = CNNNetworkHelper::getFloatData(blob);
    return std::vector<float>(buffer.get(), buffer.get() + blob->size());
}

bool isWeightableLayerQuantized(const CNNLayerPtr& weightableLayer) {
    IE_ASSERT(weightableLayer->insData.size() > 1);
    InferenceEngine::DataPtr fakeQuantizeData = weightableLayer->insData[1].lock();
    auto weightsFakeQuantizeLayer = fakeQuantizeData->getCreatorLayer().lock();
    return ((weightsFakeQuantizeLayer->type == "FakeQuantize") ||
            (fakeQuantizeData->getPrecision() == InferenceEngine::Precision::I8) ||
            (fakeQuantizeData->getPrecision() == InferenceEngine::Precision::U8));
}

bool isRealQuantizeLayer(const InferenceEngine::CNNLayerPtr& layer) {
    return ((layer->type == "Convolution") || (layer->type == "Eltwise") || (layer->type == "FullyConnected"));
}

int64_t calculateZeroPoint(float high, float low, int levels, InferenceEngine::Precision precision) {
    int64_t zepoPoint = 0;

    // Typical condition for symmetric case is low < 0, high > 0
    if (precision == InferenceEngine::Precision::I8) {
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * ((high + low) * 0.5f) / (high - low);
            zepoPoint = ceil(x);  // TODO Why not round?
        } else if (low > 0.f) {
            zepoPoint = 127 - (levels - 1);  // TODO Why not assert?
        } else if (high < 0.f) {
            zepoPoint = 127;  // TODO Why not assert?
        }
    } else if (precision == InferenceEngine::Precision::U8) {
        //  MCM team provide this formula, need check
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * low / (high - low);
            zepoPoint = ceil(x);  // TODO Why not round?
        } else if (low >= 0.f) {
            zepoPoint = 0;  // TODO Why not assert?
        } else if (high <= 0.f) {
            zepoPoint = (levels - 1);  // TODO Why not assert?
        }
    }

    return zepoPoint;
}

void reCalculateQuantizationParamsOnActivation(
    const CNNLayerPtr& quantizedLayer1, const CNNLayerPtr& quantizedLayer2, mv::QuantizationParams& outputQuantParams) {
    auto quantizationParams1 = QuantizationDetails::getDetails(*quantizedLayer1);
    auto quantizationParams2 = QuantizationDetails::getDetails(*quantizedLayer2);

    IE_ASSERT(quantizationParams1.levels == quantizationParams2.levels);
    auto levels = quantizationParams1.levels;
    IE_ASSERT(quantizationParams1.inputLowValues.size() == quantizationParams2.inputLowValues.size());

    float totalHighMax = std::max(quantizationParams1.maxOutputHigh(), quantizationParams2.maxOutputHigh());
    float totalHighLow = std::min(quantizationParams1.minOutputLow(), quantizationParams2.minOutputLow());

    int64_t zepoPoint = calculateZeroPoint(totalHighMax, totalHighLow, levels, InferenceEngine::Precision::U8);
    double scale = static_cast<double>((totalHighMax - totalHighLow) / (levels - 1));
    outputQuantParams = {{zepoPoint}, {scale}, {-inf}, {inf}};
}

void calculateOutputScalesAndZeroPoint(const CNNLayerPtr& fakeQuantizeLayer, std::vector<int64_t>& zeroPoints,
    std::vector<double>& scales, bool mergeInOne) {
    auto quantizationParams = QuantizationDetails::getDetails(*fakeQuantizeLayer);
    auto levels = quantizationParams.levels;

    if (quantizationParams.outputLowValues.size() != quantizationParams.outputHighValues.size()) {
        THROW_IE_EXCEPTION << "Unsupported case, we expect same size for outputLow and outputHigh. Layer "
                           << fakeQuantizeLayer->name;
    }

    if (mergeInOne) {
        // NOTE: Now, this branch using only for activation flow. MCM expects U8 activations
        float outputLowMin = quantizationParams.minOutputLow();
        float outputHighMax = quantizationParams.maxOutputHigh();

        int64_t zepoPoint = calculateZeroPoint(outputHighMax, outputLowMin, levels, InferenceEngine::Precision::U8);

        scales.push_back(static_cast<double>((outputHighMax - outputLowMin) / (levels - 1)));
        zeroPoints.push_back(static_cast<int64_t>(zepoPoint));
    } else {
        // NOTE: Now, this branch using only for weights. MCM expects U8 weights
        for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
            float ol = quantizationParams.outputLowValues[i];
            float oh = quantizationParams.outputHighValues[i];
            scales.push_back(static_cast<double>((oh - ol) / (levels - 1)));

            // re-calculate ZP for weights, we use I8 for weights
            int64_t zepoPoint = calculateZeroPoint(oh, ol, levels, InferenceEngine::Precision::U8);
            zeroPoints.push_back(static_cast<int64_t>(zepoPoint));
        }
    }
}

void fillQuntizationActivationParams(const CNNLayerPtr& quantizedLayer, mv::QuantizationParams& outputQuantParams) {
    std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*quantizedLayer);
    CNNLayerPtr fakeQuantizeLayer;
    // WA for case, when we should attach FQ params to not real quantized layer
    if ((!isRealQuantizeLayer(quantizedLayer) && children.size() != 1) || (children.size() == 0)) {
        outputQuantParams = initialQuantParams;
        return;
    }

    if (isPostOp(children.front())) {
        children = CNNNetworkHelper::getChildren(*(children.front()));
    }

    if (children.size() == 0) {
        outputQuantParams = initialQuantParams;
        return;
    }

    if (children.size() > 1) {
        THROW_IE_EXCEPTION << "Unsupported case, we expect only one child";
    }

    if (children.front()->type == "FakeQuantize") {
        fakeQuantizeLayer = std::dynamic_pointer_cast<InferenceEngine::QuantizeLayer>(children[0]);
    } else {
        outputQuantParams = initialQuantParams;
        return;
    }

    std::vector<int64_t> zeroPoints;
    std::vector<double> scales;
    calculateOutputScalesAndZeroPoint(fakeQuantizeLayer, zeroPoints, scales, true);
    outputQuantParams = {zeroPoints, scales, {-inf}, {inf}};
}

mv::QuantizationParams fillQuantizeParamsForU8orI8weights(
    const CNNLayerPtr& weightsLayer, int levels, InferenceEngine::Precision precision) {
    IE_ASSERT(weightsLayer->type == "Const");
    auto weightsData = getBlobValue(weightsLayer);

    float weightsLowMin = *std::min_element(weightsData.begin(), weightsData.end());
    float weightsHighMax = *std::max_element(weightsData.begin(), weightsData.end());

    int64_t zepoPoint = calculateZeroPoint(weightsLowMin, weightsHighMax, levels, precision);
    mv::QuantizationParams weightsQuantParams = {{zepoPoint}, {1}, {-inf}, {inf}};

    return weightsQuantParams;
}

Blob::Ptr quantizeWeightsBlob(const CNNLayerPtr& fakeQuantizeOnWeights, InferenceEngine::Precision precision,
    mv::QuantizationParams& weightsQuantParams) {
    InferenceEngine::DataPtr convWeights = fakeQuantizeOnWeights->insData[0].lock();
    IE_ASSERT(convWeights != nullptr);
    auto convWeightsConstLayer = convWeights->getCreatorLayer().lock();
    IE_ASSERT(convWeightsConstLayer->type == "Const");

    auto weightsBlob = convWeightsConstLayer->blobs.begin()->second;
    IE_ASSERT(weightsBlob != nullptr);

    const std::vector<size_t>& originalDims = convWeightsConstLayer->outData[0]->getDims();
    const std::vector<size_t>& dims =
        originalDims.size() == 2 ? std::vector<size_t>({originalDims[0], originalDims[1], 1, 1}) : originalDims;
    if (dims.size() != 4) {
        THROW_IE_EXCEPTION << "Unexpected dimensions count " << dims.size() << " for layer '"
                           << convWeightsConstLayer->name << "'";
    }

    const auto& sourceBlobTensorDesc = weightsBlob->getTensorDesc();
    Blob::Ptr targetBlob = make_shared_blob<uint8_t>(
        TensorDesc(precision, sourceBlobTensorDesc.getDims(), sourceBlobTensorDesc.getLayout()));
    targetBlob->allocate();

    // OIHW
    const size_t outputsSize = dims[0];  // O
    const size_t inputsSize = dims[1];   // I
    const size_t H = dims[2];            // H
    const size_t W = dims[3];            // W

    auto wQuantParams = QuantizationDetails::getDetails(*fakeQuantizeOnWeights);

    const size_t HW = H * W;
    const size_t IHW = inputsSize * HW;

    auto srcData = CNNNetworkHelper::getFloatData(weightsBlob);
    auto dstBuffer = CNNNetworkHelper::getFloatData(targetBlob);

    auto scales = weightsQuantParams.getScale();
    auto zeroPoints = weightsQuantParams.getZeroPoint();

    const bool isWeightsQuantizationBroadcasted = scales.size() != outputsSize;

    if ((scales.size() != 1) && isWeightsQuantizationBroadcasted) {
        THROW_IE_EXCEPTION << "Unexpected input low values count " << scales.size() << " for " << outputsSize
                           << " channels for " << fakeQuantizeOnWeights->name;
    }

    for (size_t outputIndex = 0; outputIndex < outputsSize; ++outputIndex) {
        for (size_t inputIndex = 0; inputIndex < inputsSize; ++inputIndex) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    const size_t idx = outputIndex * IHW + inputIndex * HW + h * W + w;

                    auto scale = scales[isWeightsQuantizationBroadcasted ? 0 : outputIndex];
                    auto zeroPoint = zeroPoints[isWeightsQuantizationBroadcasted ? 0 : outputIndex];

                    if (precision == InferenceEngine::Precision::U8) {
                        uint8_t value = clamp(std::round((srcData.get()[idx] + scale * zeroPoint) / scale), 0, 255);
                        dstBuffer.get()[idx] = value;
                    } else if (precision == InferenceEngine::Precision::I8) {
                        int8_t value = clamp(std::round((srcData.get()[idx] + scale * zeroPoint) / scale), -128, 127);
                        dstBuffer.get()[idx] = value;
                    } else {
                        THROW_IE_EXCEPTION << " Unsupported weights precision ";
                    }
                }
            }
        }
    }

    CNNNetworkHelper::fillBlobByFP32(targetBlob, dstBuffer.get());
    return targetBlob;
}

Blob::Ptr calculateQuntizationWeights(const CNNLayerPtr& weightableLayer, mv::QuantizationParams& weightsQuantParams) {
    IE_ASSERT(weightableLayer->insData.size() > 1);
    InferenceEngine::DataPtr fakeQuantizeData = weightableLayer->insData[1].lock();
    auto weightsPrecision = fakeQuantizeData->getPrecision();

    //  U8/I8 weights is already quantized, we should calculate only ZP
    if ((weightsPrecision == InferenceEngine::Precision::U8) || (weightsPrecision == InferenceEngine::Precision::I8)) {
        auto convWeightsConst = fakeQuantizeData->getCreatorLayer().lock();
        Blob::Ptr weightsBlob = convWeightsConst->blobs.begin()->second;
        weightsQuantParams = fillQuantizeParamsForU8orI8weights(convWeightsConst, 256, weightsPrecision);
        return weightsBlob;
    }

    auto convWeightsFakeQuantizeLayer = fakeQuantizeData->getCreatorLayer().lock();
    IE_ASSERT(convWeightsFakeQuantizeLayer->type == "FakeQuantize");

    std::vector<int64_t> zeroPoints;
    std::vector<double> scales;
    calculateOutputScalesAndZeroPoint(convWeightsFakeQuantizeLayer, zeroPoints, scales);
    weightsQuantParams = {zeroPoints, scales, {-inf}, {inf}};

    //  now we use I8 for weights
    return quantizeWeightsBlob(convWeightsFakeQuantizeLayer, InferenceEngine::Precision::U8, weightsQuantParams);
}

std::vector<int64_t> quantizeBiases(const std::vector<double>& activationScales,
    const std::vector<double>& weightsScales, const Blob::Ptr biasBlob, mv::QuantizationParams& outputQuantParam) {
    auto biasCount = biasBlob->size();
    const bool isWeightsScalesBroadcasted = weightsScales.size() != biasCount;
    const bool isActivationScalesBroadcasted = activationScales.size() != biasCount;
    auto biasesPrecision = biasBlob->getTensorDesc().getPrecision();

    if ((weightsScales.size() != 1) && isWeightsScalesBroadcasted) {
        THROW_IE_EXCEPTION << "Unexpected input low values count " << weightsScales.size() << " for " << biasCount
                           << " channels for bias quantization";
    }

    if ((activationScales.size() != 1) && isActivationScalesBroadcasted) {
        THROW_IE_EXCEPTION << "Unexpected input high values count " << activationScales.size() << " for " << biasCount
                           << " channels for bias quantization ";
    }

    const bool isBiasScalesBroadcasted = isWeightsScalesBroadcasted && isActivationScalesBroadcasted;
    std::vector<int64_t> newBiasData(biasCount, 0);
    std::vector<double> biasScales;

    if (biasesPrecision == InferenceEngine::Precision::FP32) {
        auto biasData = biasBlob->buffer().as<float*>();
        //  ZP = 0
        //  ScaleBias = ActivationScale * WeightsScale
        for (size_t i = 0; i < biasCount; i++) {
            double activationScale = activationScales[isActivationScalesBroadcasted ? 0 : i];
            double weightsScale = weightsScales[isWeightsScalesBroadcasted ? 0 : i];
            double biasScale = activationScale * weightsScale;
            biasScales.push_back(biasScale);
            newBiasData[i] = std::round(biasData[i] / biasScale);
        }
        int64_t biasZp = 0;
        if (isBiasScalesBroadcasted) {
            biasScales.resize(1);
        }
        outputQuantParam = mv::QuantizationParams({{biasZp}, biasScales, {0}, {1}});
    }

    if (biasesPrecision == InferenceEngine::Precision::I32) {
        auto biasData = biasBlob->buffer().as<int32_t*>();
        for (size_t i = 0; i < biasCount; i++) {
            newBiasData[i] = biasData[i];
        }
        outputQuantParam = initialQuantParams;
    }

    if (biasesPrecision != InferenceEngine::Precision::FP32 && biasesPrecision != InferenceEngine::Precision::I32) {
        THROW_IE_EXCEPTION << "Unexpected biases precision";
    }

    return newBiasData;
}

}  // namespace QuantizationHelpers
}  // namespace vpu

#endif
