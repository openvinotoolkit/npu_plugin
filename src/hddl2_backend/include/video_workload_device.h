//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

// System
#include <memory>
#include <string>

// IE
#include <ie_allocator.hpp>

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
    explicit VideoWorkloadDevice(const InferenceEngine::ParamMap& paramMap, LogLevel logLvl = LogLevel::None);

    Executor::Ptr createExecutor(const NetworkDescription::Ptr& networkDescription, const Config& config) override;

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

    // TODO: it is a stub for future implementation
    // currently, nullptr is used as a signal to use InferRequest from vpux_al
    InferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& /*networkInputs*/,
                                         const InferenceEngine::OutputsDataMap& /*networkOutputs*/,
                                         const Executor::Ptr& /*executor*/, const Config& /*config*/,
                                         const std::string& /*networkName*/,
                                         const std::vector<std::shared_ptr<const ov::Node>>& /*parameters*/,
                                         const std::vector<std::shared_ptr<const ov::Node>>& /*results*/,
                                         const std::shared_ptr<InferenceEngine::IAllocator>& /*allocator*/) override {
        return nullptr;
    }

private:
    std::shared_ptr<InferenceEngine::RemoteContext> _contextPtr;
    ParsedContextParams _contextParams;

    HDDL2RemoteAllocator::Ptr _allocatorPtr = nullptr;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
    std::string _name;
};

}  // namespace hddl2
}  // namespace vpux
