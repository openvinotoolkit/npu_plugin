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

#include <memory>

#include "InferGraph.h"
#include "hddl2_graph.h"
#include "hddl2_remote_context.h"

namespace vpu {
namespace HDDL2Plugin {

class HddlUniteGraph {
public:
    using Ptr = std::shared_ptr<HddlUniteGraph>;
    explicit HddlUniteGraph(const HDDL2Graph::Ptr& graph, const HDDL2RemoteContext::Ptr& context = nullptr);
    ~HddlUniteGraph();

private:
    HddlUnite::Inference::Graph::Ptr _graphPtr = nullptr;

    /**
     * @brief Smart pointer to remote context if we are working with remote memory
     */
    HDDL2RemoteContext::Ptr _contextPtr = nullptr;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
