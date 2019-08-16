// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_preprocess_data.hpp>
#include <map>
#include <string>
#include <memory>

namespace InferenceEngine {

class SIPPPreprocEngine;

class SIPPPreprocessor {
    std::map<std::string, std::shared_ptr<SIPPPreprocEngine>> _preprocs;

public:
    SIPPPreprocessor(const InferenceEngine::BlobMap& inputs,
                     const std::map<std::string, PreProcessDataPtr>& preprocData);

    static bool useSIPP();
    static bool isApplicable(const InferenceEngine::BlobMap& inputs,
                             const std::map<std::string, PreProcessDataPtr>& preprocData,
                             InputsDataMap& networkInputs);

    void execSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs,
                                   std::map<std::string, PreProcessDataPtr>& preprocData,
                                   InferenceEngine::InputsDataMap& networkInputs,
                                   int curBatch,
                                   bool serial);
};

}  // namespace InferenceEngine
