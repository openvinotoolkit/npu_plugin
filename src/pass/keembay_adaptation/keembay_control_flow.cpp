#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void inputOutputControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void dmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void dpuControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

        MV_REGISTER_PASS(DpuControlFlows)
        .setFunc(dpuControlFlowsFcn)
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

        if(cm.isFlowAllowedAndNonExisting(inputOp, nextOp))
            cm.defineFlow(inputOp, nextOp);
    }

    auto outputOp = om.getOutput();
    auto lastDMAOp = outputOp.leftmostParent();

    //handling one level of implicit ops first
    for (auto prevOp = outputOp.leftmostParent(); prevOp != om.opEnd(); ++prevOp)
    {
        if(!prevOp->hasTypeTrait("executable"))
        {
            for (auto parent = prevOp.leftmostParent(); parent != om.opEnd(); ++parent)
            {
                if(cm.isFlowAllowedAndNonExisting(parent, outputOp) ) 
                    cm.defineFlow(parent, outputOp);
                
            }
            
        }
        else if(cm.isFlowAllowedAndNonExisting(prevOp, outputOp)) 
            cm.defineFlow(prevOp, outputOp);
        
    }

}

// This pass adds control flows relative to a DMA Task.
// Rationale: Each DMA Task should be connected via a ControlFlow to the same operations he is connected via a DataFlow
// But implicit operations (e.g. Constants, Concat, Slice etc) must be skipped and/or avoided

// NOTE: For now, only one level of implicit operations is handled. In the future we will need a recursive procedure
void dmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto dmaOps = om.getOps("DMATask");
    for(auto op : dmaOps)
    {

        //OUTPUT DATA FLOWS OF THE DMA TASK
        for(auto outputDataFlow = op.leftmostOutput(); outputDataFlow != dm.flowEnd(); ++outputDataFlow)
        {
            auto sink = outputDataFlow.sink();
            if(!sink->hasTypeTrait("executable"))
            {
                for (auto nephew = sink.leftmostChild(); nephew != om.opEnd(); ++nephew)
                {
                    if(cm.isFlowAllowedAndNonExisting(op, nephew)) 
                        cm.defineFlow(op, nephew);
                }
            }
            else if(cm.isFlowAllowedAndNonExisting(op, sink))
                cm.defineFlow(op, sink);
        };

        // INPUT DATA FLOWS OF THE DMA TASK
        for(auto inputDataFlow = op.leftmostInput(); inputDataFlow != dm.flowEnd(); ++inputDataFlow)
        {
            auto source = inputDataFlow.source();

            // For inputs, an extra check is needed when they are not executable
            // As we can't go one level up from constants (as they have no input)
            // For outputs this check is not needed because all implicit 
            // operations have at least one output (except output itself that is handled separetely)
            if(!source->hasTypeTrait("executable"))
            {   
                for (auto parent = source.leftmostParent(); parent != om.opEnd(); ++parent)
                {
                    if(cm.isFlowAllowedAndNonExisting(parent, op)) 
                        cm.defineFlow(parent, op);
                }
            }
            else if(cm.isFlowAllowedAndNonExisting(source, op))
                cm.defineFlow(source, op);
        }
    }
}

void dpuControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto dpuTask = om.getOps("DPUTask");
    for(auto op : dpuTask)
    {

        //OUTPUT DATA FLOWS OF THE DMA TASK
        for(auto outputDataFlow = op.leftmostOutput(); outputDataFlow != dm.flowEnd(); ++outputDataFlow)
        {
            auto sink = outputDataFlow.sink();
            if(!sink->hasTypeTrait("executable"))
            {
                for (auto nephew = sink.leftmostChild(); nephew != om.opEnd(); ++nephew)
                {
                    if(cm.isFlowAllowedAndNonExisting(op, nephew)) 
                        cm.defineFlow(op, nephew);
                }
            }
            else if(cm.isFlowAllowedAndNonExisting(op, sink))
                cm.defineFlow(op, sink);
        };

        // INPUT DATA FLOWS OF THE DMA TASK
        for(auto inputDataFlow = op.leftmostInput(); inputDataFlow != dm.flowEnd(); ++inputDataFlow)
        {
            auto source = inputDataFlow.source();

            // For inputs, an extra check is needed when they are not executable
            // As we can't go one level up from constants (as they have no input)
            // For outputs this check is not needed because all implicit 
            // operations have at least one output (except output itself that is handled separetely)
            if(!source->hasTypeTrait("executable"))
            {   
                for (auto parent = source.leftmostParent(); parent != om.opEnd(); ++parent)
                {
                    if(cm.isFlowAllowedAndNonExisting(parent, op)) 
                        cm.defineFlow(parent, op);
                }
            }
            else if(cm.isFlowAllowedAndNonExisting(source, op))
                cm.defineFlow(source, op);
        }
    }
    
}


