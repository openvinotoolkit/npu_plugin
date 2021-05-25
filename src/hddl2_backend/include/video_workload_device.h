//
// Copyright 2020 Intel Corporation.
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
namespace hddl2 {

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

    HDDL2RemoteAllocator::Ptr _allocatorPtr = nullptr;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
    std::string _name;
};

}  // namespace hddl2
}  // namespace vpux
