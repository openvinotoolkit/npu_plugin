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

void verifyDMASchedulingOrderFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto dmaOps = om.getOps("DMATask");

    for (auto dmaOp : dmaOps)
    {
        if(!dmaOp->hasAttr("schedulingNumber"))
        {
            continue;
        }

        // BFS to find the next DMA ops in the graph
        std::deque<mv::Data::OpListIterator> bfsOpItr;
        std::set<std::string> childDMAs;
        bfsOpItr.push_back(dmaOp);
        while (!bfsOpItr.empty()) {
            auto currentOpItr = bfsOpItr.front();
            for(auto childIt = currentOpItr.leftmostChild();
                childIt != om.opEnd(); ++childIt) {
                if (childIt->getOpType() == "DMATask") {
                    childDMAs.insert(childIt->getName());
                } else {
                    if (std::find_if(bfsOpItr.cbegin(), bfsOpItr.cend(),
                        [&childIt](mv::Data::OpListIterator it)->bool
                        {return childIt->getName() == it->getName();})
                        == bfsOpItr.cend())
                    bfsOpItr.push_back(childIt);
                }
            }
            bfsOpItr.pop_front();
        }

        auto parentSchedulingNumber = dmaOp->get<unsigned>("schedulingNumber");
        for (auto childOpName : childDMAs)
        {
            auto childOp = om.getOp(childOpName);
            if(!childOp->hasAttr("schedulingNumber"))
            {
                continue;
            }

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
