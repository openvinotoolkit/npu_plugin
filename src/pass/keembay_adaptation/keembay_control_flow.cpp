#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void inputOutputControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void dmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(DmaControlFlows)
        .setFunc(dmaControlFlowsFcn)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(InputOutputControlFlows)
        .setFunc(inputOutputControlFlowsFcn)
        .setDescription(
            ""
        );

    }

}


// This pass adds Input and Output control flows of a network.
void inputOutputControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto inputOp = om.getInput();

    //auto nextOp = inputOp.leftmostChild();
    for (auto nextOp = inputOp.leftmostChild(); nextOp != om.opEnd(); ++nextOp)
    {
        if(!nextOp->hasTypeTrait("executable"))
                continue;

        if(!cm.checkControlFlow(inputOp, nextOp))
            cm.defineFlow(inputOp, nextOp);
    }

    auto outputOp = om.getOutput();
    auto lastDMAOp = outputOp.leftmostParent();

    if(!cm.checkControlFlow(lastDMAOp, outputOp))
        cm.defineFlow(lastDMAOp, outputOp);

    // Adding the memory requirement here is not required anymore since we have explicit dealloc for last op as well.
}

// This pass adds control flows relative to a DMA Task.
// Rationale: Each DMA Task should be connected via a ControlFlow to the same operations he is connected via a DataFlow
void dmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto dmaOps = om.getOps("DMATask");
    for(auto op : dmaOps)
    {
        for(auto outputDataFlow = op.leftmostOutput(); outputDataFlow != dm.flowEnd(); ++outputDataFlow)
        {
            auto sink = outputDataFlow.sink();
            if(!sink->hasTypeTrait("executable"))
                continue;

            if(!cm.checkControlFlow(op, sink))
                cm.defineFlow(op, sink);
        }
    }
}


