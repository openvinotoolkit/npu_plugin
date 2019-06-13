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
#include <vpu/blob_reader.hpp>
#include <net_pass.h>

using namespace InferenceEngine;

namespace vpu {
namespace KmbPlugin {

ExecutableNetwork::ExecutableNetwork(ICNNNetwork &network, std::vector<DevicePtr> &devicePool,
                                     const std::map<std::string, std::string> &config) {
    _config = std::make_shared<KmbConfig>(config);
//    _config->hostLogLevel = LogLevel::Debug;
//    _config->deviceLogLevel = LogLevel::Debug;

    _log = std::make_shared<Logger>("KmbPlugin", _config->hostLogLevel, consoleOutput());

    _executor = std::make_shared<KmbExecutor>(_config->forceReset, _config->deviceLogLevel, _log);
#if 0  // there is no KMB device yet
    _device = _executor->openDevice(devicePool, _config);

    // ignore hardware optimization config for MYRIAD2, it is always disabled
    if (_device->_platform == MYRIAD_2) {
        _config->compileConfig.hwOptimization = false;
    }
#endif

    pCompiler = std::make_shared<mv::CompilationUnit>(network.getName());  // unit("testModel");

    if (pCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.";
    }

    _log->debug("CompilationUnit and model '%s' are created", pCompiler->model().getName());
    bool ti_proc_ok = !NetPass::CombineRNNSeq(network) ? NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";

#if 1
    compileMcm(
        network,
        *(_config),
        *pCompiler,
        std::make_shared<Logger>("GraphCompiler", _config->hostLogLevel, consoleOutput()));
#else
    auto compiledGraph = compileNetwork(
        network,
        static_cast<Platform>(_device->_platform),
        _config->compileConfig,
        std::make_shared<Logger>("GraphCompiler", _config->hostLogLevel, consoleOutput()));

    char networkName[1024] = {};
    network.getName(networkName, sizeof(networkName));

    _graphBlob = std::move(compiledGraph->blob);
    _stagesMetaData = std::move(compiledGraph->stagesMeta);

    _inputInfo  = std::move(compiledGraph->inputInfo);
    _outputInfo = std::move(compiledGraph->outputInfo);

    if (!_device->isBooted()) {
        return;
    }

    char networkName[1024] = {};
    network.getName(networkName, sizeof(networkName));
    _executor->allocateGraph(_device, _graphDesc, _graphBlob, compiledGraph->blobHeader, compiledGraph->numActiveStages, networkName);
    if (_config->exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor(
                TargetDeviceInfo::name(TargetDevice::eKMB));
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
#endif
}

ExecutableNetwork::ExecutableNetwork(const std::string &blobFilename,
                           std::vector<DevicePtr> &devicePool,
                           const std::map<std::string, std::string> &config) {
    _config = std::make_shared<KmbConfig>(config, ConfigMode::RUNTIME_MODE);

    _log = std::make_shared<Logger>("KmbPlugin", _config->hostLogLevel, consoleOutput());

    _executor = std::make_shared<KmbExecutor>(_config->forceReset, _config->deviceLogLevel, _log);
#if 0
    _device = _executor->openDevice(devicePool, _config);

    // ignore hardware optimization config for MYRIAD2, it is always disabled
    if (_device->_platform == MYRIAD_2) {
        _config->compileConfig.hwOptimization = false;
    }
#endif
    std::ifstream blobFile(blobFilename, std::ios::binary);
    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();

#if 0
    if (!_device->isBooted()) {
        return;
    }
#endif

    // TODO: better name
    char networkName[1024] = "importedNetwork";

    KmbBlob blobReader(blobContentString.data(), blobContentString.size());

    this->_networkInputs  = blobReader.getNetworkInputs();
    this->_networkOutputs = blobReader.getNetworkOutputs();
    std::size_t numStages = blobReader.getStageCount();
    auto blobHeader = blobReader.getHeader();


    _inputInfo  = blobReader.getInputInfo();
    _outputInfo = blobReader.getOutputInfo();

    _executor->allocateGraph(_device,
                             _graphDesc,
                             _graphBlob,
                             blobHeader,
                             numStages,
                             networkName);

    _stagesMetaData.resize(numStages);
    for (auto &meta : _stagesMetaData) {
        meta.stageName = meta.stageType = meta.layerName = meta.layerType = "UNKNOWN";
        meta.status = InferenceEngineProfileInfo::LayerStatus::EXECUTED;
    }

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
