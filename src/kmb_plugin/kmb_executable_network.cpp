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

#include <algorithm>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <kmb_executable_network.h>
#include <net_pass.h>
#include "vpu/kmb_plugin_config.hpp"

using namespace InferenceEngine;

namespace vpu {
namespace KmbPlugin {

void ExecutableNetwork::ConfigureExecutor() {
    // TODO: better name
    const char networkName[1024] = "Network";

    if (_config.exclusiveAsyncRequests()) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("KMB");
    }
    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

void ExecutableNetwork::LoadBlob() {
    _executor->allocateGraph(_graphBlob);
    _networkInputs  = _executor->getNetworkInputs();
    _networkOutputs = _executor->getNetworkOutputs();
}

ExecutableNetwork::ExecutableNetwork(ICNNNetwork &network, const KmbConfig& config) : _config(config) {
    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());
    _executor = std::make_shared<KmbExecutor>(_config);

#ifdef ENABLE_MCM_COMPILER
    pCompiler = std::make_shared<mv::CompilationUnit>(network.getName());  // unit("testModel");

    if (pCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.";
    }

    _logger->debug("CompilationUnit and model '%s' are created", pCompiler->model().getName());
    bool ti_proc_ok = !NetPass::CombineRNNSeq(network) ? NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";


        compileMcm(network, _config, *pCompiler, _graphBlob);
        auto parsedConfig = _config.getParsedConfig();
        if (parsedConfig[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] == CONFIG_VALUE(YES)) {
            LoadBlob();
            ConfigureExecutor();
        }
#else
    UNUSED(network);
#endif
    _supportedMetrics = {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)
    };
}

ExecutableNetwork::ExecutableNetwork(const std::string &blobFilename, const KmbConfig& config) : _config(config) {
    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());
    _executor = std::make_shared<KmbExecutor>(_config);
    std::ifstream blobFile(blobFilename, std::ios::binary);
    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(_graphBlob));
    LoadBlob();
    ConfigureExecutor();
}

void ExecutableNetwork::GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    UNUSED(resp);
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(4u));
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
}


}  // namespace KmbPlugin
}  // namespace vpu
