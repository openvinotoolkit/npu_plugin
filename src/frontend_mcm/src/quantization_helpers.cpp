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

#include <precision_utils.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "ie_utils.hpp"

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
    return ((layer->type == "ReLU") || (layer->type == "Clamp") || (layer->type == "ReorgYolo"));
}

bool isRealQuantizeLayer(const InferenceEngine::CNNLayerPtr& layer) {
    return ((layer->type == "Convolution") || (layer->type == "Eltwise") || (layer->type == "FullyConnected"));
}

int64_t calculateZeroPoint(float high, float low, int levels, InferenceEngine::Precision precision) {
    int64_t zeroPoint = 0;

    // Typical condition for symmetric case is low < 0, high > 0
    if (precision == InferenceEngine::Precision::I8) {
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * ((high + low) * 0.5f) / (high - low);
            zeroPoint = round(x);
        } else if (low > 0.f) {
            zeroPoint = 127 - (levels - 1);  // TODO Why not assert?
        } else if (high < 0.f) {
            zeroPoint = 127;  // TODO Why not assert?
        }
    } else if (precision == InferenceEngine::Precision::U8) {
        //  MCM team provide this formula, need check
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * low / (high - low);
            zeroPoint = round(x);
        } else if (low >= 0.f) {
            zeroPoint = 0;  // TODO Why not assert?
        } else if (high <= 0.f) {
            zeroPoint = (levels - 1);  // TODO Why not assert?
        }
    }

    return zeroPoint;
}

mv::QuantizationParams calculateOutputScalesAndZeroPoint(const CNNLayerPtr& fakeQuantizeLayer, bool mergeInOne) {
    auto quantizationParams = QuantizationDetails::getDetails(*fakeQuantizeLayer);
    int levels = quantizationParams.levels;

    if (quantizationParams.outputLowValues.size() != quantizationParams.outputHighValues.size()) {
        THROW_IE_EXCEPTION << "Unsupported case, we expect same size for outputLow and outputHigh. Layer "
                           << fakeQuantizeLayer->name;
    }

    std::vector<int64_t> zeroPoints;
    std::vector<double> scales;
    std::vector<double> mins;
    std::vector<double> maxs;
    mv::QuantizationParams outputQuantParams = initialQuantParams;

    if (mergeInOne) {
        // NOTE: Now, this branch using only for activation flow. MCM expects U8 activations
        float outputLowMin = quantizationParams.minOutputLow();
        float outputHighMax = quantizationParams.maxOutputHigh();

        int64_t zepoPoint = calculateZeroPoint(outputHighMax, outputLowMin, levels, InferenceEngine::Precision::U8);

        scales.push_back(static_cast<double>((outputHighMax - outputLowMin) / (levels - 1)));
        zeroPoints.push_back(static_cast<int64_t>(zepoPoint));
        mins.push_back(outputLowMin);
        maxs.push_back(outputHighMax);
    } else {
        // NOTE: Now, this branch using only for weights. MCM expects U8 weights
        int64_t avgZeroPoints = 0;
        for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
            float ol = quantizationParams.outputLowValues[i];
            float oh = quantizationParams.outputHighValues[i];

            // re-calculate ZP for weights, we use U8 for weights
            avgZeroPoints += calculateZeroPoint(oh, ol, levels, InferenceEngine::Precision::U8);
        }
        avgZeroPoints = std::round(static_cast<double>(avgZeroPoints) / quantizationParams.outputLowValues.size());

        for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
            float ol = quantizationParams.outputLowValues[i];
            float oh = quantizationParams.outputHighValues[i];

            float zpl = oh * avgZeroPoints / (avgZeroPoints - (levels - 1));
            float zph = ol - ol * (levels - 1) / avgZeroPoints;

            ol = std::min(ol, zpl);
            oh = std::max(oh, zph);

            scales.push_back(static_cast<double>((oh - ol) / (levels - 1)));
            zeroPoints.push_back(avgZeroPoints);
            mins.push_back(ol);
            maxs.push_back(oh);
        }
    }
    outputQuantParams = {zeroPoints, scales, mins, maxs};
    return outputQuantParams;
}

void fillQuntizationActivationParams(const CNNLayerPtr& quantizedLayer, mv::QuantizationParams& outputQuantParams) {
    std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*quantizedLayer);
    CNNLayerPtr fakeQuantizeLayer;
    // WA for case, when we should attach FQ params to not real quantized layer
    if ((!isRealQuantizeLayer(quantizedLayer) && children.size() != 1) || (children.size() == 0)) {
        outputQuantParams = initialQuantParams;
        return;
    }

    while (isPostOp(children.front())) {
        children = CNNNetworkHelper::getChildren(*(children.front()));
        if (children.size() == 0) {
            outputQuantParams = initialQuantParams;
            return;
        }
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

    outputQuantParams = calculateOutputScalesAndZeroPoint(fakeQuantizeLayer, true);
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

    if (biasesPrecision == InferenceEngine::Precision::FP32 || biasesPrecision == InferenceEngine::Precision::FP16) {
        ie::Blob::Ptr biasBlobFp32 = toFP32(biasBlob);
        auto biasData = biasBlobFp32->buffer().as<float*>();
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

    return newBiasData;
}

}  // namespace QuantizationHelpers
}  // namespace vpu

#endif
