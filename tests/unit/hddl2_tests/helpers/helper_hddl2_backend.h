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

#include <memory>

#include "hddl2_backend.h"
namespace vpux {

class HDDL2Backend_Helper {
public:
    using Ptr = std::shared_ptr<HDDL2Backend_Helper>;
    HDDL2Backend_Helper();
    const std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& map);

protected:
    HDDL2::HDDL2Backend::Ptr _backend;
};

//------------------------------------------------------------------------------
inline HDDL2Backend_Helper::HDDL2Backend_Helper() : _backend(std::make_shared<HDDL2::HDDL2Backend>()) {}
inline const std::shared_ptr<Device> HDDL2Backend_Helper::getDevice(const InferenceEngine::ParamMap& map) {
    return std::make_shared<Device>(_backend->getDevice(map), nullptr);
}
}

