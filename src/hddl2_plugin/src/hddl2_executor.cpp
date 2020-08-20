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

#include "hddl2_executor.h"

#include <ie_preprocess.hpp>
#include <ie_remote_context.hpp>

#include "hddl2_exceptions.h"
#include "hddl2_metrics.h"
#include "hddl2_remote_context.h"
#include "hddl_unite/hddl2_unite_graph.h"

namespace vpux {
namespace HDDL2 {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
/**
 * @brief If service is not available, we cannot create any executor
 */
static bool isServiceAvailable(const vpu::Logger::Ptr& logger) {
    if (vpu::HDDL2Plugin::HDDL2Metrics::isServiceAvailable()) {
        logger->debug(SERVICE_AVAILABLE.c_str());
        return true;
    } else {
        logger->debug(SERVICE_NOT_AVAILABLE.c_str());
        return false;
    }
}

static vpu::HDDL2Plugin::HDDL2RemoteContext::Ptr castIEContextToHDDL2(const IE::RemoteContext::Ptr& ieContext) {
    vpu::HDDL2Plugin::HDDL2RemoteContext::Ptr pluginContext = nullptr;

    if (ieContext != nullptr) {
        pluginContext = std::dynamic_pointer_cast<vpu::HDDL2Plugin::HDDL2RemoteContext>(ieContext);
        if (pluginContext == nullptr) {
            THROW_IE_EXCEPTION << FAILED_CAST_CONTEXT;
        }
    }
    return pluginContext;
}

//------------------------------------------------------------------------------
vpux::HDDL2::HDDL2Executor::Ptr HDDL2Executor::prepareExecutor(const vpux::NetworkDescription::Ptr& networkDesc,
    const vpu::HDDL2Config& config, const InferenceEngine::RemoteContext::Ptr& ieContextPtr) {
    auto logger = std::make_shared<vpu::Logger>("Executor", config.logLevel(), vpu::consoleOutput());
    vpux::HDDL2::HDDL2Executor::Ptr executor = nullptr;
    if (!isServiceAvailable(logger)) {
        logger->warning(EXECUTOR_NOT_CREATED.c_str());
        return nullptr;
    }

    auto context = castIEContextToHDDL2(ieContextPtr);

    try {
        executor = std::make_shared<vpux::HDDL2::HDDL2Executor>(networkDesc, config, context);
    } catch (const IE::details::InferenceEngineException& exception) {
        if (exception.hasStatus() && exception.getStatus() == IE::StatusCode::NETWORK_NOT_LOADED) {
            logger->error(FAILED_LOAD_NETWORK.c_str());
        } else {
            logger->error("%s%s", EXECUTOR_NOT_CREATED.c_str(), std::string("\nERROR: ") + exception.what());
        }
    } catch (const std::exception& exception) {
        logger->error("%s%s", EXECUTOR_NOT_CREATED.c_str(), std::string("\nERROR: ") + exception.what());
    }
    return executor;
}

HDDL2Executor::HDDL2Executor(const vpux::NetworkDescription::Ptr& network, const vpu::HDDL2Config& config,
    vpu::HDDL2Plugin::HDDL2RemoteContext::Ptr context)
    : _network(network),
      _context(context),
      _config(config),
      _logger(std::make_shared<vpu::Logger>("ExecutableNetwork", config.logLevel(), vpu::consoleOutput())) {
    loadGraphToDevice();
}

void HDDL2Executor::setup(const InferenceEngine::ParamMap& params) {
    UNUSED(params);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2Executor::push(const InferenceEngine::BlobMap& inputs) { UNUSED(inputs); }

void HDDL2Executor::push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) {
    UNUSED(inputs);
    UNUSED(preProcMap);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2Executor::pull(InferenceEngine::BlobMap& outputs) {
    UNUSED(outputs);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

bool HDDL2Executor::isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const {
    UNUSED(preProcessInfo);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> HDDL2Executor::getLayerStatistics() {
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

InferenceEngine::Parameter HDDL2Executor::getParameter(const std::string& paramName) const {
    UNUSED(paramName);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2Executor::loadGraphToDevice() {
    if (_context == nullptr) {
        _uniteGraphPtr =
            std::make_shared<vpu::HDDL2Plugin::HddlUniteGraph>(_network, _config.device_id(), _config.logLevel());
    } else {
        _uniteGraphPtr = std::make_shared<vpu::HDDL2Plugin::HddlUniteGraph>(_network, _context, _config.logLevel());
    }
}
}  // namespace HDDL2
}  // namespace vpux
