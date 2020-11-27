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
#include "hddl2_remote_allocator.h"
// Subplugin
#include "vpux.hpp"
// Low-level
#include "APICommon.h"
#include "WorkloadContext.h"

namespace vpux {
namespace HDDL2 {

class ParsedContextParams final {
public:
    explicit ParsedContextParams(const InferenceEngine::ParamMap& paramMap);
    InferenceEngine::ParamMap getParamMap() const;

    WorkloadID getWorkloadId() const;

private:
    InferenceEngine::ParamMap _paramMap;

    WorkloadID _workloadId;
};

//------------------------------------------------------------------------------
/**
 * @brief Context dependent device
 */
class VideoWorkloadDevice final : public IDevice {
public:
    explicit VideoWorkloadDevice(const InferenceEngine::ParamMap& paramMap, const VPUXConfig& config = {});
    Executor::Ptr createExecutor(const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

    std::shared_ptr<Allocator> getAllocator() const override {
        return nullptr;
    }
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const override;
    std::string getName() const override {
        return _name;
    }
    const ParsedContextParams& getContextParams() const {
        return _contextParams;
    }
    HddlUnite::WorkloadContext::Ptr getUniteContext() const {
        return _workloadContext;
    }

private:
    InferenceEngine::RemoteContext::Ptr _contextPtr;
    ParsedContextParams _contextParams;

    vpu::HDDL2Plugin::HDDL2RemoteAllocator::Ptr _allocatorPtr = nullptr;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
    std::string _name;
};

}  // namespace HDDL2
}  // namespace vpux
