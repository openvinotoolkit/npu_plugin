//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

// System
#include <memory>

// Plugin
#include "infer_data_adapter.h"
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
                            const HddlUnite::WorkloadContext::Ptr& workloadContext, const Config& config);

    /** @brief Create HddlUnite graph object using specific device. If empty, use all available devices */
    explicit HddlUniteGraph(const vpux::NetworkDescription::CPtr& network, const std::string& deviceID,
                            const Config& config);

    ~HddlUniteGraph();
    void InferAsync(const InferDataAdapter::Ptr& data) const;

private:
    HddlUnite::Inference::Graph::Ptr _uniteGraphPtr = nullptr;
    Logger _logger;
};

}  // namespace hddl2
}  // namespace vpux
