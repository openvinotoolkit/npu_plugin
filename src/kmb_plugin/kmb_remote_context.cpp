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

#include <kmb_remote_context.h>
#include <kmb_remote_blob.h>

#include <memory>
#include <string>

using namespace vpu::KmbPlugin;

static std::map<std::string, int> deviceIdMapping = {
    { "VPU-0", 0 },
};

KmbContextParams::KmbContextParams(const InferenceEngine::ParamMap& params)
    : _paramMap(params),
      _deviceId(-1),
      _deviceIdStr("") {
    if (_paramMap.empty()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbContextParams: Param map for context is empty.";
    }

    auto deviceIdIter = _paramMap.find(InferenceEngine::KMB_PARAM_KEY(DEVICE_ID));
    if (deviceIdIter == _paramMap.end()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbContextParams: Param map does not contain device ID.";
    }
    _deviceIdStr = deviceIdIter->second.as<std::string>();

    auto devMappingIter = deviceIdMapping.find(_deviceIdStr);
    if (devMappingIter == deviceIdMapping.end()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbContextParams: Device ID " << _deviceIdStr << " is invalid.";
    }
    _deviceId = devMappingIter->second;
}

InferenceEngine::ParamMap KmbContextParams::getParamMap() const { return _paramMap; }

int KmbContextParams::getDeviceId() const { return _deviceId; }

std::string KmbContextParams::getDeviceIdStr() const { return _deviceIdStr; }

KmbRemoteContext::KmbRemoteContext(const InferenceEngine::ParamMap& ctxParams, const KmbConfig& config)
     : _config(config),
      _contextParams(ctxParams),
      _logger(std::make_shared<Logger>("KmbRemoteContext", config.logLevel(), consoleOutput())),
      _deviceId(_contextParams.getDeviceId()) {
    _allocatorPtr = std::make_shared<KmbAllocator>(_deviceId);
}

InferenceEngine::RemoteBlob::Ptr KmbRemoteContext::CreateBlob(
    const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params) noexcept {
    try {
        auto smart_this = shared_from_this();
    } catch (...) {
        _logger->warning("Please use smart pointer to context instead of instance of class\n");
        return nullptr;
    }
    try {
        return std::make_shared<KmbRemoteBlob>(tensorDesc, shared_from_this(), params, _config);
    } catch (const std::exception& ex) {
        _logger->warning("Incorrect parameters for CreateBlob call.\n"
                         "Please make sure remote memory fd is correct.\nError: %s\n",
            ex.what());
        return nullptr;
    }
}

std::string KmbRemoteContext::getDeviceName() const noexcept {
    return "KMB." + _contextParams.getDeviceIdStr();
}

KmbAllocator::Ptr KmbRemoteContext::getAllocator() { return _allocatorPtr; }

InferenceEngine::ParamMap KmbRemoteContext::getParams() const { return _contextParams.getParamMap(); }

int KmbRemoteContext::getDeviceId() const noexcept { return _deviceId; }

KmbContextParams KmbRemoteContext::getContextParams() const { return _contextParams; }
