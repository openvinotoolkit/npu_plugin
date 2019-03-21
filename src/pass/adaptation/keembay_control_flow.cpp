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

        MV_REGISTER_PASS(InputOutputControlFlows)
        .setFunc(inputOutputControlFlowsFcn)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(DmaControlFlows)
        .setFunc(dmaControlFlowsFcn)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(DeallocationControlFlows)
        .setFunc(deallocationControlFlowsFcn)
        .setDescription(
            ""
        );

    }

}



void inputOutputControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    //auto inputOp = om.getInput();
    //cm.defineFlow(inputOp, inputOp.leftmostChild());

    auto outputOp = om.getOp("Output_0");
    auto lastDMAOp = outputOp.leftmostParent();
    cm.defineFlow(lastDMAOp, outputOp);
}

void dmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto dmaOps = om.getOps("DMATask");
    for(auto op : dmaOps)
    {
        if(op->get<mv::DmaDirection>("direction") == mv::DDR2CMX)
        {
            for(auto outputDataFlow = op.leftmostOutput(); outputDataFlow != dm.flowEnd(); ++outputDataFlow)
            {
                auto sink = outputDataFlow.sink();
                if(sink->getOpType() != "Deallocate")
                    cm.defineFlow(op, sink);
            }
        }
    }
}

void deallocationControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto deallocateOps = om.getOps("Deallocate");
    for(auto op : deallocateOps)
    {
        auto controlInputOp = cm.switchContext(op).leftmostParent();
        for(auto outputDataFlow = om.switchContext(controlInputOp).leftmostOutput(); outputDataFlow != dm.flowEnd(); ++outputDataFlow)
        {
            auto sink = outputDataFlow.sink();
            if(sink->getOpType() != "Deallocate")
                cm.defineFlow(op, sink);
        }
    }
}
