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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "InferGraph.h"
#include "hddl2_remote_context.h"
#include "ie_blob.h"
#include "ie_input_info.hpp"

namespace vpu {
namespace HDDL2Plugin {

class HddlUniteInferData {
public:
    using Ptr = std::shared_ptr<HddlUniteInferData>;

    explicit HddlUniteInferData(const HDDL2RemoteContext::Ptr& remoteContext = nullptr);

    void prepareInput(const std::string& inputName, const InferenceEngine::Blob::Ptr& blob);

    void prepareOutput(const std::string& outputName, const InferenceEngine::Blob::Ptr& blob);

    static HddlUnite::Inference::Precision convertIEPrecision(const InferenceEngine::Precision& precision);

    HddlUnite::Inference::InferData::Ptr& getHddlUniteInferData() { return _inferDataPtr; }

private:
    void createRemoteDesc(const bool isInput, const std::string& name, const InferenceEngine::Blob::Ptr& blob);

    void createLocalDesc(const bool isInput, const std::string& name, const InferenceEngine::Blob::Ptr& blob);

    std::vector<HddlUnite::Inference::AuxBlob::Type> _auxBlob;
    HddlUnite::Inference::InferData::Ptr _inferDataPtr = nullptr;

    bool isVideoWorkload = false;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
