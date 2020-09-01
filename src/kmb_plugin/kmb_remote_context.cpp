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

#include "kmb_remote_context.h"

#include <memory>
#include <string>

#include "kmb_remote_blob.h"

using namespace vpu::KmbPlugin;

KmbRemoteContext::KmbRemoteContext(
    const InferenceEngine::ParamMap& params, const KmbConfig& config, const std::shared_ptr<vpux::Device>& device)
    : _config(config),
      _params(params),
      _logger(std::make_shared<Logger>("KmbRemoteContext", config.logLevel(), consoleOutput())),
      _device(device) {
    if (_device == nullptr) {
        THROW_IE_EXCEPTION << "Cannot define a remote context with a device";
    }
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
        return std::make_shared<KmbRemoteBlob>(
            tensorDesc, shared_from_this(), params, _config, _device->getAllocator());
    } catch (const std::exception& ex) {
        _logger->warning("Incorrect parameters for CreateBlob call.\n"
                         "Please make sure remote memory fd is correct.\nError: %s\n",
            ex.what());
        return nullptr;
    }
}

std::string KmbRemoteContext::getDeviceName() const noexcept { return "KMB." + _device->getName(); }

InferenceEngine::ParamMap KmbRemoteContext::getParams() const { return _params; }
