//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

// System
#include <atomic>
#include <mutex>

// Plugin
#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux_remote_context.h"

// Low-level
#include "hddl_unite/hddl2_unite_graph.h"

namespace vpux {
namespace hddl2 {

class HDDL2Executor final : public Executor {
public:
    using Ptr = std::shared_ptr<HDDL2Executor>;
    using CPtr = std::shared_ptr<const HDDL2Executor>;

    HDDL2Executor(const HDDL2Executor& ex);
    explicit HDDL2Executor(const vpux::NetworkDescription::CPtr& network, const Config& config,
                           const std::shared_ptr<vpux::Allocator>& allocator,
                           const HddlUnite::WorkloadContext::Ptr& workloadContext);
    HDDL2Executor& operator=(const HDDL2Executor& ex) = delete;
    static HDDL2Executor::Ptr prepareExecutor(const vpux::NetworkDescription::Ptr& networkDesc, const Config& config,
                                              const std::shared_ptr<vpux::Allocator>& allocator = nullptr,
                                              const HddlUnite::WorkloadContext::Ptr& workloadContext = nullptr);

    void setup(const InferenceEngine::ParamMap& params) override;

    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    void pull(InferenceEngine::BlobMap& outputs) override;

    bool isPreProcessingSupported(const PreprocMap& preProcMap) const override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;
    void push(const InferenceEngine::BlobMap& inputs) override;

    Executor::Ptr clone() const override;

private:
    void loadGraphToDevice();

private:
    Config _config;
    Logger _logger;

    NetworkDescription::CPtr _network;

    HddlUniteGraph::Ptr _uniteGraphPtr = nullptr;
    InferDataAdapter::Ptr _inferDataPtr = nullptr;

    // Variables below might be not required for executor
    std::shared_ptr<vpux::Allocator> _allocatorPtr = nullptr;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
    HddlUnite::RemoteMemoryDesc _remoteMemoryDesc = {0, 0, 0, 0};

    // TODO [Track number: S#37397] [Workaround] Avoid allocation inferData each time. If size of inputs has changed,
    // need  to recreating (not implemented yet)
    std::once_flag _onceFlagInferData;

    std::mutex _uniteGraphMapMutex;
    const size_t _baseExecutorId;

    static std::atomic<size_t> _executorIdCounter;
    static std::map<size_t, std::weak_ptr<HddlUniteGraph>> _uniteGraphMap;
};

}  // namespace hddl2
}  // namespace vpux
