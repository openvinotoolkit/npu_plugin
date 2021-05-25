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
// Plugin
#include "infer_data_adapter.h"
#include "vpux_config.hpp"
#include "vpux_remote_context.h"
// Subplugin
#include "vpux.hpp"
// Low-level
#include "InferGraph.h"

namespace vpux {
namespace hddl2 {

class HddlUniteGraph final {
public:
    using Ptr = std::shared_ptr<HddlUniteGraph>;
    using CPtr = std::shared_ptr<const HddlUniteGraph>;

    /**  @brief Create HddlUnite graph object using context to specify which devices to use */
    explicit HddlUniteGraph(const vpux::NetworkDescription::CPtr& network,
                            const HddlUnite::WorkloadContext::Ptr& workloadContext,
                            const vpux::VPUXConfig& config = {});

    /** @brief Create HddlUnite graph object using specific device. If empty, use all available devices */
    explicit HddlUniteGraph(const vpux::NetworkDescription::CPtr& network, const std::string& deviceID = "",
                            const vpux::VPUXConfig& config = {});

    ~HddlUniteGraph();
    void InferAsync(const InferDataAdapter::Ptr& data) const;

private:
    HddlUnite::Inference::Graph::Ptr _uniteGraphPtr = nullptr;
    const vpu::Logger::Ptr _logger;
};

}  // namespace hddl2
}  // namespace vpux
