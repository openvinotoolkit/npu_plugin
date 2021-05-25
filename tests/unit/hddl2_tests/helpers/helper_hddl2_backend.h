//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <memory>

#include "hddl2_backend.h"
namespace vpux {

class HDDL2Backend_Helper {
public:
    using Ptr = std::shared_ptr<HDDL2Backend_Helper>;
    HDDL2Backend_Helper();
    const std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& map);

protected:
    IEngineBackendPtr _backend;
};

//------------------------------------------------------------------------------
inline HDDL2Backend_Helper::HDDL2Backend_Helper() : _backend(getLibFilePath("hddl2_backend")) {}
inline const std::shared_ptr<Device> HDDL2Backend_Helper::getDevice(const InferenceEngine::ParamMap& map) {
    return std::make_shared<Device>(_backend->getDevice(map), _backend);
}
}

