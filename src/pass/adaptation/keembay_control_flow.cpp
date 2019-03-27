#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void inputOutputControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void dmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void deallocationControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(DeallocationControlFlows)
        .setFunc(deallocationControlFlowsFcn)
        .setDescription(
            ""
        );

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
    auto nextOp = inputOp.leftmostChild();
    if(!cm.checkControlFlow(inputOp, nextOp))
        cm.defineFlow(inputOp, nextOp);

    auto outputOp = om.getOutput();
    auto lastDMAOp = outputOp.leftmostParent();
    if(!cm.checkControlFlow(lastDMAOp, outputOp))
        cm.defineFlow(lastDMAOp, outputOp);
}

// This pass adds Control flows relative to a DeallocateTask
// Rational: A Deallocate task has to be connected via a Control flow to the next operations in the graph (coming in DataFlow order)

// So we take the leftMostParent of the deallocTask(in terms of ControlFlow), take the output of this op (in terms of DataFlow)
// verify that it is not a dealloc task and connect it.
void deallocationControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto deallocateOps = om.getOps("Deallocate");
    for(auto op : deallocateOps)
    {
        auto controlInputOp = cm.switchContext(op).leftmostParent();
        auto dataInputOp = dm.switchContext(controlInputOp);
        for(auto outputDataFlow = dataInputOp.leftmostOutput(); outputDataFlow != dm.flowEnd(); ++outputDataFlow)
        {
            auto sink = outputDataFlow.sink();
            if(sink->getOpType() != "Deallocate")
                if(!cm.checkControlFlow(op, sink))
                    cm.defineFlow(op, sink);
        }
    }
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
            if(!cm.checkControlFlow(op, sink))
                cm.defineFlow(op, sink);
        }
    }
}


