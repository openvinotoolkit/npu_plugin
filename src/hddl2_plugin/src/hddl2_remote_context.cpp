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

// System
#include <memory>
#include <string>
// Plugin
#include "hddl2_params.hpp"
#include "hddl2_remote_context.h"
#include "subplugin/hddl2_remote_blob.h"
// Subplugin
#include <subplugin/hddl2_context_device.h>

using namespace vpu::HDDL2Plugin;

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
HDDL2RemoteContext::HDDL2RemoteContext(const InferenceEngine::ParamMap& paramMap, const vpu::HDDL2Config& config)
    : _config(config),
      _logger(std::make_shared<Logger>("VPUXRemoteContext", config.logLevel(), consoleOutput())),
      _paramMap(paramMap) {
    // TODO There should be searching for corresponding device
    _devicePtr = std::make_shared<vpux::HDDL2::HDDLUniteContextDevice>(paramMap, config);
}

IE::RemoteBlob::Ptr HDDL2RemoteContext::CreateBlob(
    const IE::TensorDesc& tensorDesc, const IE::ParamMap& params) noexcept {
    try {
        auto smart_this = shared_from_this();
    } catch (...) {
        _logger->warning("Please use smart ptr to context instead of instance of class\n");
        return nullptr;
    }
    try {
        auto allocator = _devicePtr->getAllocator();
        return std::make_shared<HDDL2RemoteBlob>(tensorDesc, shared_from_this(), allocator, params, _config);
    } catch (const std::exception& ex) {
        _logger->warning("Incorrect parameters for CreateBlob call.\n"
                         "Please make sure remote memory is correct.\nError: %s\n",
            ex.what());
        return nullptr;
    }
}

std::string HDDL2RemoteContext::getDeviceName() const noexcept { return "VPUX." + _devicePtr->getName(); }

IE::ParamMap HDDL2RemoteContext::getParams() const { return _paramMap; }

std::shared_ptr<vpux::IDevice> HDDL2RemoteContext::getDevice() const { return _devicePtr; }
