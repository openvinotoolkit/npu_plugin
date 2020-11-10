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

#include <ie_allocator.hpp>
#include <memory>
#include <string>
#include <vpual_config.hpp>
#include <vpux.hpp>

namespace vpux {

class VpualDevice final : public IDevice {
public:
    VpualDevice(const std::string& name);
    std::shared_ptr<Allocator> getAllocator() const override;

    std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

    std::string getName() const override;

private:
    std::shared_ptr<Allocator> _allocator;
    const std::string _name;
    // TODO: config is used in executor only
    // it makes sense to store it only there
    VpualConfig _config;
};

}  // namespace vpux
