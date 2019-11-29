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

#include <cpp_interfaces/ie_executor_manager.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "kmb_async_infer_request.h"
#include "kmb_config.h"
#include "kmb_executable_network.h"
#include "kmb_executor.h"
#include "kmb_infer_request.h"
#include "kmb_parser.hpp"

namespace vpu {
namespace KmbPlugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<ExecutableNetwork> Ptr;

    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork& network, const KmbConfig& config);

    explicit ExecutableNetwork(const std::string& blobFilename, const KmbConfig& config);

    ~ExecutableNetwork() {
        try {
            if (_executor) {
                _executor->deallocateGraph();
            }
        } catch (...) {
            std::cerr << "ERROR ~ExecutableNetwork():\n"
                      << "Some errors occurred during the calling of the deallocateGraph() method";
        }
    }

    void GetMetric(const std::string& name, InferenceEngine::Parameter& result,
        InferenceEngine::ResponseDesc* resp) const override;

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<KmbInferRequest>(networkInputs, networkOutputs, _stagesMetaData, _config, _executor);
    }

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr& asyncRequest) override {
        auto syncRequestImpl =
            std::make_shared<KmbInferRequest>(_networkInputs, _networkOutputs, _stagesMetaData, _config, _executor);
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto taskExecutorGetResult = getNextTaskExecutor();
        auto asyncTreadSafeImpl = std::make_shared<KmbAsyncInferRequest>(
            syncRequestImpl, _taskExecutor, taskExecutorGetResult, _callbackExecutor, _logger);
        asyncRequest.reset(new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(
                               asyncTreadSafeImpl),
            [](InferenceEngine::IInferRequest* p) {
                p->Release();
            });
        asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    }

    void Export(const std::string& modelFileName) override {
        std::ofstream modelFile(modelFileName, std::ios::out | std::ios::binary);

        if (modelFile.is_open()) {
            modelFile.write(_graphBlob.data(), _graphBlob.size());
        } else {
            THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export";
        }
    }

    void GetMappedTopology(
        std::map<std::string, std::vector<InferenceEngine::PrimitiveInfo::Ptr>>& deployedTopology) override {
        UNUSED(deployedTopology);
        THROW_IE_EXCEPTION << "GetMappedTopology is not implemented\n";
    }

private:
    Logger::Ptr _logger;
#ifdef ENABLE_MCM_COMPILER
    std::shared_ptr<mv::CompilationUnit> pCompiler;
#endif
    KmbExecutorPtr _executor;
    std::vector<char> _graphBlob;
    std::vector<StageMetaInfo> _stagesMetaData;
    KmbConfig _config;
    std::vector<std::string> _supportedMetrics;

    const size_t _maxTaskExecutorGetResultCount = 1;
    std::queue<std::string> _taskExecutorGetResultIds;

    void ConfigureExecutor(const std::string& networkName);
    void LoadBlob();

    InferenceEngine::ITaskExecutor::Ptr getNextTaskExecutor() {
        std::string id = _taskExecutorGetResultIds.front();

        _taskExecutorGetResultIds.pop();
        _taskExecutorGetResultIds.push(id);

        InferenceEngine::ExecutorManager* executorManager = InferenceEngine::ExecutorManager::getInstance();
        InferenceEngine::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

        return taskExecutor;
    }
};

}  // namespace KmbPlugin
}  // namespace vpu
