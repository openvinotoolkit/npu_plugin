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
// Plugin
#include "infer_data_adapter.h"
#include "vpux_config.hpp"
#include "vpux_remote_context.h"
// Subplugin
#include "vpux.hpp"
// Low-level
#include "InferGraph.h"

namespace vpu {
namespace HDDL2Plugin {

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
    const Logger::Ptr _logger;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
