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
    KmbExecutorPtr _executor;
    InferenceEngine::Layout _deviceLayout;
    std::vector<StageMetaInfo> _stagesMetaData;
    KmbConfig _config;

protected:
    InferenceEngine::BlobMap _custom_inputs;
    InferenceEngine::BlobMap _custom_outputs;
    void checkBlobs();
    void dumpInputBlobHelper(const InferenceEngine::Blob::Ptr& inputBlobPtr, const std::string& dst);
    void dumpOutputBlobHelper(const InferenceEngine::Blob::Ptr& outputBlobPtr, const std::string& dst);

public:
    typedef std::shared_ptr<KmbInferRequest> Ptr;

    explicit KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
        const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<StageMetaInfo>& blobMetaData,
        const KmbConfig& kmbConfig, const KmbExecutorPtr& executor);

    void InferImpl() override;
    void InferAsync();
    void GetResult();

    void GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

    void SetBlob(const char* name, const InferenceEngine::Blob::Ptr& data) override;
    void GetBlob(const char* name, InferenceEngine::Blob::Ptr& data) override;

    void Infer() override;

private:
    Logger::Ptr _logger;
};

}  // namespace KmbPlugin
}  // namespace vpu
