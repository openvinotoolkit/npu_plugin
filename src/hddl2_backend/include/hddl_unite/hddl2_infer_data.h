//
// Copyright 2020 Intel Corporation.
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

// System
#include <map>
#include <memory>
#include <string>
#include <vector>
// IE
#include "ie_blob.h"
#include "ie_input_info.hpp"
#include "ie_preprocess_data.hpp"
// Plugin
#include "blob_descriptor.h"
#include "vpux_remote_context.h"
// Low-level
#include "InferData.h"
#include "InferGraph.h"

namespace vpu {
namespace HDDL2Plugin {

/**
 * Carries information necessary for invoking infer request on Unite
 */
class HddlUniteInferData final {
public:
    using Ptr = std::shared_ptr<HddlUniteInferData>;

    explicit HddlUniteInferData(const HddlUnite::WorkloadContext::Ptr& workloadContext = nullptr,
        const InferenceEngine::ColorFormat colorFormat = InferenceEngine::ColorFormat::BGR,
        const size_t numOutputs = 1);

    void setPreprocessFlag(const bool preprocessingRequired);

    // TODO Now similar to prepareUniteOutput, make one for all?
    void prepareUniteInput(const InferenceEngine::Blob::CPtr& blob, const InferenceEngine::DataPtr& desc);
    void prepareUniteOutput(const InferenceEngine::DataPtr& desc);

    HddlUnite::Inference::InferData::Ptr& getHddlUniteInferData() { return _inferDataPtr; }
    void waitInferDone() const;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getHDDLUnitePerfCounters() const;

    /**
     * @brief Wait when inference is done and get result from HddlUnite
     */
    std::string getOutputData(const std::string& outputName);

private:
    const int _asyncInferenceWaitTimeoutMs = 10000;
    std::vector<HddlUnite::Inference::AuxBlob::Type> _auxBlob;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
    HddlUnite::Inference::InferData::Ptr _inferDataPtr = nullptr;

    std::map<std::string, BlobDescriptor::Ptr> _inputs;
    std::map<std::string, BlobDescriptor::Ptr> _outputs;

    const bool _haveRemoteContext;
    bool _needUnitePreProcessing;

    // TODO [Workaround] Avoid allocation buffer each time
    std::once_flag _onceFlagInputAllocations;

    std::vector<std::string> _onceFlagOutputAllocations;

    HddlUnite::Inference::InferData::ProfileData _profileData = {};
    InferenceEngine::ColorFormat _graphColorFormat = InferenceEngine::ColorFormat::BGR;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
