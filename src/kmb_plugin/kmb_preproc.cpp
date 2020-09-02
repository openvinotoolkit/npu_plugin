// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_preproc.hpp"

#include <map>
#include <memory>
#include <string>

#include "ie_macro.hpp"

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_pool.hpp"
#endif

namespace InferenceEngine {
namespace KmbPreproc {

#if defined(__arm__) || defined(__aarch64__)
static bool supported(ResizeAlgorithm interp, ColorFormat inFmt) {
    return (interp == RESIZE_BILINEAR) && (inFmt == ColorFormat::NV12);
}
#endif

bool isApplicable(const InferenceEngine::BlobMap& inputs, const std::map<std::string, PreProcessDataPtr>& preprocData,
    InputsDataMap& networkInputs) {
#if defined(__arm__) || defined(__aarch64__)
    if (inputs.size() != 1 || preprocData.empty()) return false;

    for (auto& input : inputs) {
        const auto& blobName = input.first;
        auto it = preprocData.find(blobName);
        if (it != preprocData.end()) {
            const auto& preprocInfo = networkInputs[blobName]->getPreProcess();
            if (!supported(preprocInfo.getResizeAlgorithm(), preprocInfo.getColorFormat())) {
                return false;
            }
        }
    }
    return true;
#else
    UNUSED(inputs);
    UNUSED(preprocData);
    UNUSED(networkInputs);
    return false;
#endif
}

void execDataPreprocessing(InferenceEngine::BlobMap& inputs, std::map<std::string, PreProcessDataPtr>& preprocData,
    InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
    unsigned int lpi, const std::string& preprocPoolId, const int deviceId, Path ppPath) {
#if defined(__arm__) || defined(__aarch64__)
    IE_ASSERT(numShaves > 0 && numShaves <= 16)
        << "KmbPreproc::execDataPreprocessing "
        << "attempt to set invalid number of shaves: " << numShaves << ", valid numbers are from 1 to 16";
    preprocPool().execDataPreprocessing(
        {inputs, preprocData, networkInputs, out_format}, numShaves, lpi, ppPath, preprocPoolId, deviceId);
#else
    UNUSED(inputs);
    UNUSED(preprocData);
    UNUSED(networkInputs);
    UNUSED(out_format);
    UNUSED(numShaves);
    UNUSED(lpi);
    UNUSED(preprocPoolId);
    UNUSED(ppPath);
    UNUSED(deviceId);
    THROW_IE_EXCEPTION << "VPUAL is disabled. Used only for arm";
#endif
}
}  // namespace KmbPreproc
}  // namespace InferenceEngine
