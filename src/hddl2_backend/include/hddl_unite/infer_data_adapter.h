//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

// System
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
// IE
#include "ie_blob.h"
#include "ie_input_info.hpp"
#include "ie_preprocess_data.hpp"
// Plugin
#include "blob_descriptor_adapter.h"
#include "vpux_remote_context.h"
// Low-level
#include "InferData.h"
#include "InferGraph.h"

namespace vpux {
namespace hddl2 {

/**
 * @brief Carries information necessary for invoking infer request on HddlUnite
 * @details Wrap HddlUnite::InferData method
 */
class InferDataAdapter final {
public:
    InferDataAdapter() = delete;
    InferDataAdapter(const InferDataAdapter&) = delete;
    InferDataAdapter(const InferDataAdapter&&) = delete;
    InferDataAdapter& operator=(const InferDataAdapter&) = delete;
    InferDataAdapter& operator=(const InferDataAdapter&&) = delete;
    explicit InferDataAdapter(const vpux::NetworkDescription::CPtr& networkDescription,
                              const HddlUnite::WorkloadContext::Ptr& workloadContext = nullptr,
                              const InferenceEngine::ColorFormat graphColorFormat = InferenceEngine::ColorFormat::BGR);

public:
    using Ptr = std::shared_ptr<InferDataAdapter>;
    void setPreprocessFlag(const bool preprocessingRequired);

    void prepareUniteInput(const InferenceEngine::Blob::CPtr& blob, const std::string& inputName,
                           const InferenceEngine::ColorFormat inputColorFormat);

    void waitInferDone() const;

    HddlUnite::Inference::InferData::Ptr& getHDDLUniteInferData() {
        return _inferDataPtr;
    }
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getHDDLUnitePerfCounters() const;

    std::string getOutputData(const std::string& outputName);

private:
    void createInferData();

private:
    const vpux::NetworkDescription::CPtr& _networkDescription;

    const int _asyncInferenceWaitTimeoutMs = 30000;
    std::vector<HddlUnite::Inference::AuxBlob::Type> _auxBlob;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
    HddlUnite::Inference::InferData::Ptr _inferDataPtr = nullptr;
    InferenceEngine::ColorFormat _graphColorFormat;

    std::map<std::string, BlobDescriptorAdapter::Ptr> _inputs;
    std::map<std::string, BlobDescriptorAdapter::Ptr> _outputs;

    const bool _haveRemoteContext;
    bool _needUnitePreProcessing;

    HddlUnite::Inference::InferData::ProfileData _profileData = {};

private:  // Workarounds
    // TODO Use maxRoiNum
    const size_t maxRoiNum = 1;
};

}  // namespace hddl2
}  // namespace vpux
