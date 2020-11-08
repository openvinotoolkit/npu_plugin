// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <unordered_set>

namespace vpu {
namespace details {

namespace ie = InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

/**
 * @def THROW_VPU_LPT_EXCEPTION
 * @brief A macro used to throw the exception with a notable description for low precision transformations
 */
#define THROW_VPU_LPT_EXCEPTION(layer)                                                                     \
    THROW_IE_EXCEPTION << "Exception during low precision transformation for " << layer.type << " layer '" \
                       << layer.name << "'. "

/**
 * @brief CNNNetworkHelper class encapsulates manipulations with CNN Network.
 */
class CNNNetworkHelper {
public:
    static ie::Blob::Ptr makeNewBlobPtr(const ie::TensorDesc& desc);

    static void updateBlobs(const ie::CNNLayer& quantizeLayer, int constLayerIndex, const std::vector<float>& values);

    // return true if at least one child uses layer on weights
    static bool onWeights(const ie::CNNLayer& layer);

    static bool onConstWeightsPath(const ie::CNNLayer& quantize);

    static size_t getOutputChannelsCount(const ie::CNNLayer& layer, bool isOnWeights = false);

    static std::shared_ptr<float> getFloatData(const ie::Blob::Ptr& srcBlob);

    static bool isBlobPrecisionSupported(const ie::Precision precision);

    static void fillBlobByFP32(ie::Blob::Ptr& dstBlob, float value);

    static void fillBlobByFP32(ie::Blob::Ptr& dstBlob, const float* srcData);

    static ie::CNNLayerPtr getParent(
        const ie::CNNLayer& layer, const size_t index = 0, const std::string& ignoreLayerType = "");

    static std::vector<ie::CNNLayerPtr> getParents(
        const ie::CNNLayer& layer, const std::string& exceptionLayerName = "");

    static std::vector<ie::CNNLayerPtr> getParentsRecursivelyExceptTypes(const ie::CNNLayer& layer,
        const std::unordered_set<std::string>& exceptionLayerTypes = {}, const int portIndex = -1);

    static std::vector<ie::CNNLayerPtr> getChildren(
        const ie::CNNLayer& layer, const std::string& exceptionLayerName = "");

private:
    // 1  - on weights
    // 0  - weightable layer was not found
    // -1 - on activations
    static int onWeightsInDepth(const ie::CNNLayer& layer);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace vpu
