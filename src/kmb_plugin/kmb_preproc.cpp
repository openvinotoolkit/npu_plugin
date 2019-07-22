// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_preproc.hpp"
#include <map>
#include <string>

#ifdef ENABLE_VPUAL
#include "kmb_preproc_gapi.hpp"
#endif

namespace InferenceEngine {

#ifdef ENABLE_VPUAL
SIPPPreprocessor::SIPPPreprocessor(const InferenceEngine::BlobMap& inputs,
                                   const std::map<std::string, PreProcessData>& preprocData) {
    for (auto& input : inputs) {
        auto it = preprocData.find(input.first);
        if (it != preprocData.end()) {
            _preprocs.emplace(input.first, new SIPPPreprocEngine);
        }
    }
}

static bool supported(ResizeAlgorithm interp, ColorFormat inFmt) {
    return (interp == RESIZE_BILINEAR) &&
           (inFmt == ColorFormat::NV12);
}

bool SIPPPreprocessor::isApplicable(const InferenceEngine::BlobMap& inputs,
                  const std::map<std::string, PreProcessData>& preprocData,
                  InputsDataMap& networkInputs) {
    for (auto& input : inputs) {
        const auto& blobName = input.first;
        auto it = preprocData.find(blobName);
        if (it != preprocData.end()) {
            const auto& preprocInfo = networkInputs[blobName]->getPreProcess();
            if (!supported(preprocInfo.getResizeAlgorithm(),
                           preprocInfo.getColorFormat())) {
                return false;
            }
        }
    }
    return true;
}

void SIPPPreprocessor::execSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs,
                                                 std::map<std::string, PreProcessData>& preprocData,
                                                 InputsDataMap& networkInputs,
                                                 int curBatch,
                                                 bool serial) {
    for (auto& input : inputs) {
        const auto& blobName = input.first;
        auto it = preprocData.find(blobName);
        if (it != preprocData.end()) {
            const auto& preprocInfo = networkInputs.at(blobName)->getPreProcess();
            _preprocs.at(blobName)->preprocWithSIPP(preprocData.at(blobName).getRoiBlob(),
                                                    input.second,
                                                    preprocInfo.getResizeAlgorithm(),
                                                    preprocInfo.getColorFormat(),
                                                    serial,
                                                    curBatch);
        }
    }
}
#else
SIPPPreprocessor::SIPPPreprocessor(const InferenceEngine::BlobMap&,
                                   const std::map<std::string, PreProcessData>&) {
    THROW_IE_EXCEPTION << "Error: SIPPPreprocessor::SIPPPreprocessor() "
                       << "should never be called when ENABLE_VPUAL is OFF";
}

bool SIPPPreprocessor::isApplicable(const InferenceEngine::BlobMap&,
                  const std::map<std::string, PreProcessData>&,
                  InputsDataMap&) {
    return false;
}

void SIPPPreprocessor::execSIPPDataPreprocessing(InferenceEngine::BlobMap&,
                                                 std::map<std::string, PreProcessData>&,
                                                 InputsDataMap&, int, bool) {
    THROW_IE_EXCEPTION << "Error: SIPPPreprocessor::execSIPPDataPreprocessing() "
                       << "should never be called when ENABLE_VPUAL is OFF";
}
#endif
}  // namespace InferenceEngine
