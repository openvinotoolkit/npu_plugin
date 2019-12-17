//
// Copyright 2019 Intel Corporation.
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

#pragma once
#include <ie_layers_internal.hpp>
#include <low_precision_transformations/network_helper.hpp>
#include <low_precision_transformations/quantization_details.hpp>
#include <vector>

#ifdef ENABLE_MCM_COMPILER

#include <include/mcm/op_model.hpp>

namespace vpu {

namespace KmbPlugin {

namespace KmbQuantizationHelpers {

bool isPostOp(const InferenceEngine::CNNLayerPtr& layer);
std::vector<float> getBlobValue(const InferenceEngine::CNNLayerPtr& constantLayer);
bool isWeightableLayerQuantized(const InferenceEngine::CNNLayerPtr& weightableLayer);
bool isRealQuantizeLayer(const InferenceEngine::CNNLayerPtr& layer);

void calculateOutputScalesAndZeroPoint(const InferenceEngine::CNNLayerPtr& fakeQuantizeLayer,
    std::vector<int64_t>& zeroPoints, std::vector<double>& scales, bool mergeInOne = false);

void fillQuntizationActivationParams(
    const InferenceEngine::CNNLayerPtr& quantizedLayer, mv::QuantizationParams& outputQuantParams);

InferenceEngine::Blob::Ptr calculateQuntizationWeights(
    const InferenceEngine::CNNLayerPtr& quantizedLayer, mv::QuantizationParams& weightsQuantParams);

// for symmetric case only, using mcm logic
int64_t calculateZeroPoint(float high, float low, int levels, InferenceEngine::Precision precision);

void reCalculateQuantizationParamsOnActivation(const InferenceEngine::CNNLayerPtr& quantizedLayer1,
    const InferenceEngine::CNNLayerPtr& quantizedLayer2, mv::QuantizationParams& outputQuantParams);

InferenceEngine::Blob::Ptr calculateQuntizationWeights(
    const InferenceEngine::CNNLayerPtr& weightableLayer, mv::QuantizationParams& weightsQuantParams);

mv::QuantizationParams fillQuantizeParamsForU8orI8weights(
    const InferenceEngine::CNNLayerPtr& weightableLayer, int levels, InferenceEngine::Precision precision);

std::vector<int64_t> quantizeBiases(const std::vector<double>& activationScales,
    const std::vector<double>& weightsScales, const InferenceEngine::Blob::Ptr biasBlob,
    mv::QuantizationParams& outputQuantParam);

}  // namespace KmbQuantizationHelpers
}  // namespace KmbPlugin
}  // namespace vpu

#endif
