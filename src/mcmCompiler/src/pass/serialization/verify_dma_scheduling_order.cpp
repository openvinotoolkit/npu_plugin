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

#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"

static void verifyDMASchedulingOrderFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(VerifyDMASchedulingOrder)
        .setFunc(verifyDMASchedulingOrderFcn)
        .setDescription(
            "Verify if after all scheduler and barrier optimization passes"
            "the DMA execution order is still correct"
        );
    }
}

void verifyDMASchedulingOrderFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto dmaOps = om.getOps("DMATask");

    for (auto dmaOp : dmaOps)
    {
        // BFS to find the next DMA ops in the graph
        std::queue<mv::Data::OpListIterator> bfsOpItr;
        std::vector<mv::Data::OpListIterator> childDMAs;
        bfsOpItr.push(dmaOp);
        while (!bfsOpItr.empty()) {
            auto currentOpItr = bfsOpItr.front();
            for(auto childIt = currentOpItr.leftmostChild();
                childIt != om.opEnd(); ++childIt) {
                if (childIt->getOpType() == "DMATask") {
                    childDMAs.push_back(childIt);
                } else {
                    bfsOpItr.push(childIt);
                }
            }
            bfsOpItr.pop();
        }

        auto parentSchedulingNumber = dmaOp->get<unsigned>("schedulingNumber");
        for (auto childOp : childDMAs)
        {
            if (childOp->getOpType() == "DMATask" &&
                childOp->get<unsigned>("schedulingNumber") < parentSchedulingNumber)
            {
                mv::Logger::log(mv::Logger::MessageType::Error, "VerifyDMASchedulingOrder",
                    "Following subsequent DMAs " + dmaOp->getName() + " -> " + childOp->getName() +
                    " have incorrect schedule numbers " + std::to_string(parentSchedulingNumber) +
                    " -> " + std::to_string(childOp->get<unsigned>("schedulingNumber")) + ".");
            }
        }
    }
}
