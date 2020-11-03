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
// Plugin
#include "vpux.hpp"
#include "vpux_config.hpp"
#include "vpux_remote_context.h"
// Low-level
#include "hddl_unite/hddl2_unite_graph.h"

namespace vpux {
namespace HDDL2 {
class HDDL2Executor final : public Executor {
public:
    using Ptr = std::shared_ptr<HDDL2Executor>;
    using CPtr = std::shared_ptr<const HDDL2Executor>;

    HDDL2Executor(const HDDL2Executor& ex);
    explicit HDDL2Executor(const vpux::NetworkDescription::CPtr& network, const vpux::VPUXConfig& config,
        const std::shared_ptr<vpux::Allocator>& allocator, const HddlUnite::WorkloadContext::Ptr& workloadContext);
    HDDL2Executor& operator=(const HDDL2Executor& ex) = delete;
    static HDDL2Executor::Ptr prepareExecutor(const vpux::NetworkDescription::Ptr& networkDesc,
        const VPUXConfig& config, const std::shared_ptr<vpux::Allocator>& allocator = nullptr,
        const HddlUnite::WorkloadContext::Ptr& workloadContext = nullptr);

    void setup(const InferenceEngine::ParamMap& params) override;

    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    void pull(InferenceEngine::BlobMap& outputs) override;

    bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;
    void push(const InferenceEngine::BlobMap& inputs) override;

    Executor::Ptr clone() const override;

private:
    void loadGraphToDevice();

private:
    vpux::VPUXConfig _config;
    const vpu::Logger::Ptr _logger;

    NetworkDescription::CPtr _network;

    vpu::HDDL2Plugin::HddlUniteGraph::Ptr _uniteGraphPtr = nullptr;
    vpu::HDDL2Plugin::HddlUniteInferData::Ptr _inferDataPtr = nullptr;

    // Variables below might be not required for executor
    std::shared_ptr<vpux::Allocator> _allocatorPtr = nullptr;
    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;

    // TODO [Track number: S#37397] [Workaround] Avoid allocation inferData each time. If size of inputs is changed,
    // need  to recreating (not implemented yet)
    std::once_flag _onceFlagInferData;
};
}  // namespace HDDL2
}  // namespace vpux
