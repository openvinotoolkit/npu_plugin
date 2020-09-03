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

#include <ie_common.h>

#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/perf_report.hpp>

#include "kmb_config.h"
#include "kmb_executor.h"
#include "kmb_preproc.hpp"

namespace vpu {
namespace KmbPlugin {

class KmbInferRequest : public InferenceEngine::InferRequestInternal {
    std::shared_ptr<vpux::Executor> _executor;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;
    std::vector<StageMetaInfo> _stagesMetaData;
    KmbConfig _config;
    const std::string _netUniqueId;
    const int _deviceId;
    // the buffer is used when non-shareable memory passed for preprocessing
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _prepprocBuffer;
    Logger::Ptr _logger;

public:
    using Ptr = std::shared_ptr<KmbInferRequest>;

    explicit KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
        const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<vpu::StageMetaInfo>& blobMetaData,
        const KmbConfig& kmbConfig, const std::shared_ptr<vpux::Executor>& executor,
        const std::shared_ptr<InferenceEngine::IAllocator>& allocator, const std::string& netName, const int deviceId);

    void InferImpl() override;
    void InferAsync();
    void GetResult();

    void GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

protected:
    void execPreprocessing(InferenceEngine::BlobMap& inputs);
    void relocationAndExecKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
        InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
        unsigned int lpi);
    virtual void execKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
        std::map<std::string, InferenceEngine::PreProcessDataPtr>& preprocData,
        InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
        unsigned int lpi);
};

}  // namespace KmbPlugin
}  // namespace vpu
