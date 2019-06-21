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

#include <map>
#include <string>
#include <vector>
#include <memory>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

#include <vpu/utils/logger.hpp>

#include "kmb_executor.h"
#include "kmb_config.h"

namespace vpu {
namespace KmbPlugin {

class KmbInferRequest : public InferenceEngine::InferRequestInternal {
    KmbExecutorPtr _executor;
    InferenceEngine::Layout _deviceLayout;
    Logger::Ptr _log;
    std::vector<StageMetaInfo> _stagesMetaData;
    std::shared_ptr<KmbConfig> _config;

    const DataInfo _inputInfo;
    const DataInfo _outputInfo;

    std::vector<uint8_t> resultBuffer;
    std::vector<uint8_t> inputBuffer;

public:
    typedef std::shared_ptr<KmbInferRequest> Ptr;

    explicit KmbInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                InferenceEngine::OutputsDataMap networkOutputs,
                                DataInfo& inputInfo,
                                DataInfo& outputInfo,
                                const std::vector<StageMetaInfo> &blobMetaData,
                                const std::shared_ptr<KmbConfig> &kmbConfig,
                                const Logger::Ptr &log,
                                const KmbExecutorPtr &executor);

    void InferImpl() override;
    void InferAsync();
    void GetResult();

    void
    GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;
};

}  // namespace KmbPlugin
}  // namespace vpu
