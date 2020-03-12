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

#include <ie_blob.h>

#include <ie_input_info.hpp>
#include <ie_preprocess_data.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "InferGraph.h"
#include "blob_descriptor.h"
#include "hddl2_remote_context.h"

namespace vpu {
namespace HDDL2Plugin {

class HddlUniteInferData {
public:
    using Ptr = std::shared_ptr<HddlUniteInferData>;

    explicit HddlUniteInferData(
        const bool& needPreProcessing = false, const HDDL2RemoteContext::Ptr& remoteContext = nullptr);

    void prepareInput(const InferenceEngine::Blob::Ptr& blob, const InferenceEngine::InputInfo::Ptr& info);
    void prepareOutput(const InferenceEngine::Blob::Ptr& blob, const InferenceEngine::DataPtr& desc);

    HddlUnite::Inference::InferData::Ptr& getHddlUniteInferData() { return _inferDataPtr; }

    std::string getOutputData(const std::string& outputName);

private:
    std::vector<HddlUnite::Inference::AuxBlob::Type> _auxBlob;
    HddlUnite::Inference::InferData::Ptr _inferDataPtr = nullptr;

    std::map<std::string, BlobDescriptor::Ptr> _inputs;
    std::map<std::string, BlobDescriptor::Ptr> _outputs;

    bool isVideoWorkload = false;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
