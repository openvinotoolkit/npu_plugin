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

#include <kmb_executable_network.h>
#include <net_pass.h>

using namespace InferenceEngine;

namespace vpu {
namespace KmbPlugin {

ExecutableNetwork::ExecutableNetwork(ICNNNetwork &network, const std::map<std::string, std::string> &config) {
    _config = std::make_shared<KmbConfig>(config);

    _log = std::make_shared<Logger>("KmbPlugin", _config->hostLogLevel, consoleOutput());

    _executor = std::make_shared<KmbExecutor>(_log, _config);

#ifdef ENABLE_MCM_COMPILER
    pCompiler = std::make_shared<mv::CompilationUnit>(network.getName());  // unit("testModel");

    if (pCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.";
    }

    _log->debug("CompilationUnit and model '%s' are created", pCompiler->model().getName());
    bool ti_proc_ok = !NetPass::CombineRNNSeq(network) ? NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";

    compileMcm(
        network,
        *(_config),
        *pCompiler,
        _graphBlob);
#else
    UNUSED(network);
#endif
}

ExecutableNetwork::ExecutableNetwork(const std::string &blobFilename, const std::map<std::string, std::string> &config) {
    _config = std::make_shared<KmbConfig>(config, ConfigMode::RUNTIME_MODE);

    _log = std::make_shared<Logger>("KmbPlugin", _config->hostLogLevel, consoleOutput());

    _executor = std::make_shared<KmbExecutor>(_log, _config);
    std::ifstream blobFile(blobFilename, std::ios::binary);
    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(_graphBlob));

    // TODO: better name
    const char networkName[1024] = "importedNetwork";

    _executor->allocateGraph(_graphBlob, &networkName[0]);

    _networkInputs  = _executor->getNetworkInputs();
    _networkOutputs = _executor->getNetworkOutputs();

    if (_config->exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("KMB");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

}  // namespace KmbPlugin
}  // namespace vpu
