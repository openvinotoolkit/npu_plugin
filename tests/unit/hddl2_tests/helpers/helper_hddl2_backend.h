//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
    EngineBackend _backend;
};

//------------------------------------------------------------------------------
inline HDDL2Backend_Helper::HDDL2Backend_Helper() : _backend(getLibFilePath("vpux_hddl2_backend")) {}
inline const std::shared_ptr<Device> HDDL2Backend_Helper::getDevice(const InferenceEngine::ParamMap& map) {
    return _backend.getDevice(map);
}
}
