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
// suppress 'error: ‘inf’ defined but not used'
// static double inf = std::numeric_limits<double>::infinity();
// TODO remove when it is fixed in mcmCompiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <include/mcm/op_model.hpp>
#pragma GCC diagnostic pop

namespace vpu {

namespace KmbPlugin {

namespace KmbQuantizationHelpers {

bool isPostOp(const InferenceEngine::CNNLayerPtr& layer);
std::vector<float> getBlobValue(const InferenceEngine::CNNLayerPtr& constantLayer);
bool isWeightableLayerQuantized(const InferenceEngine::CNNLayerPtr& weightableLayer);
bool isRealQuantizeLayer(const InferenceEngine::CNNLayerPtr& layer);

void calculateOutputScalesAndZeroPoint(const InferenceEngine::CNNLayerPtr &fakeQuantizeLayer,
                                       std::vector<int64_t> &zeroPoints,
                                       std::vector<double> &scales, bool mergeInOne = false);

void fillQuntizationActivationParams(const InferenceEngine::CNNLayerPtr& quantizedLayer, mv::QuantizationParams &outputQuantParams);

InferenceEngine::Blob::Ptr calculateQuntizationWeights(const InferenceEngine::CNNLayerPtr& quantizedLayer,
                                                       mv::QuantizationParams &weightsQuantParams);

// for symmetric case only, using mcm logic
int64_t calculateZeroPoint(float high, float low, InferenceEngine::Precision precision);

void reCalculateQuantizationParamsOnActivation(const InferenceEngine::CNNLayerPtr& quantizedLayer1,
                                               const InferenceEngine::CNNLayerPtr& quantizedLayer2, mv::QuantizationParams &outputQuantParams);

InferenceEngine::Blob::Ptr calculateQuntizationWeights(const InferenceEngine::CNNLayerPtr& weightableLayer,
                                                       mv::QuantizationParams &weightsQuantParams);

mv::QuantizationParams fillQuantizeParamsForU8orI8weights(const InferenceEngine::CNNLayerPtr& weightableLayer, InferenceEngine::Precision precision);


}  // namespace KmbQuantizationHelpers
}  // namespace KmbPlugin
}  // namespace vpu

#endif
