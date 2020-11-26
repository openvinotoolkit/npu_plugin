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

#pragma once

// System
#include <memory>
#include <string>
// IE
#include "ie_allocator.hpp"
// Plugin
#include "vpux.hpp"

namespace vpux {
namespace HDDL2 {
/**
 * @brief General device, for ImageWorkload.
 * If specific name not provided, device selection will be postponed until inference
 */
class HDDLUniteDevice final : public IDevice {
public:
    explicit HDDLUniteDevice(const std::string& name = "");
    std::shared_ptr<Allocator> getAllocator() const override {
        return _allocatorPtr;
    }
    std::string getName() const override {
        return _name;
    }

    Executor::Ptr createExecutor(const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

private:
    std::shared_ptr<Allocator> _allocatorPtr = nullptr;
    const std::string _name;
};

}  // namespace HDDL2
}  // namespace vpux
