// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_preproc.hpp"

#include <map>
#include <memory>
#include <string>

#include "kmb_allocator.h"

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_pool.hpp"
#endif

namespace InferenceEngine {

namespace SippPreproc {
bool useSIPP() {
#if defined(__arm__) || defined(__aarch64__)
    const bool USE_SIPP = [](const char* str) -> bool {
        std::string var(str ? str : "");
        return var == "Y" || var == "YES" || var == "ON" || var == "1";
    }(std::getenv("USE_SIPP"));

    return USE_SIPP;
#else
    return false;
#endif
}

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

// TODO: Drop SIPP from the name (let it be KMB or maybe GAPI if
// there's a need to filter this out)
void execSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs, std::map<std::string, PreProcessDataPtr>& preprocData,
    InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
    unsigned int lpi, Path ppPath) {
#if defined(__arm__) || defined(__aarch64__)
    IE_ASSERT(numShaves > 0 && numShaves <= 16)
        << "SippPreproc::execSIPPDataPreprocessing "
        << "attempt to set invalid number of shaves for SIPP: " << numShaves << ", valid numbers are from 1 to 16";
    sippPreprocPool().execSIPPDataPreprocessing(
        {inputs, preprocData, networkInputs, out_format}, numShaves, lpi, ppPath);
#else
    UNUSED(inputs);
    UNUSED(preprocData);
    UNUSED(networkInputs);
    UNUSED(out_format);
    UNUSED(numShaves);
    UNUSED(lpi);
    UNUSED(ppPath);
    THROW_IE_EXCEPTION << "VPUAL is disabled. Used only for arm";
#endif
}
}  // namespace SippPreproc
}  // namespace InferenceEngine
