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
    auto lastOpBeforeLastDma = lastDMAOp.leftmostParent();

    if(!cm.checkControlFlow(lastDMAOp, outputOp))
        cm.defineFlow(lastDMAOp, outputOp);
    if(!cm.checkControlFlow(lastOpBeforeLastDma, lastDMAOp)) {

         /* 
          Add the memory requirment of the DPU task -> DMA CMX2DDR direction (last DPU task) by adding a "MemoryRequirment" attribute to the flow in the MCM task graph.
          The memory requirment is defined as the output tensor (N*W*H*C) * dataType.
          This is required for the max topological cut pass.
        */
        auto flowIt = cm.defineFlow(lastOpBeforeLastDma, lastDMAOp);

        int memoryRequirement = 1;
        auto dType = flowIt.source()->getOutputTensor()[0]->get<mv::DType>("dType").getSizeInBits();
             
        for (unsigned int i = 0; i < flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
            memoryRequirement = flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

        memoryRequirement = memoryRequirement * dType/8;
        flowIt->set<int>("MemoryRequirement", memoryRequirement);
        flowIt->set<bool>("PositiveMemory", true); 

    }
}

// This pass adds Control flows relative to a DeallocateTask
// A deallocate task must happen after the operation in which the data is involved. Basically all siblings in DataFlow Context (inflows)
// A deallocate task must also happen after the operation that allocated the data. (inflows)

// A deallocate task must also happen before the his nieces (outflow) in both DataFlow and ControlFlow contex.
void deallocationControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto deallocateOps = om.getOps("Deallocate");
    for(auto op : deallocateOps)
    {
        auto parentOp = op.leftmostParent();
        auto flowIt = cm.defineFlow(parentOp, op);

        /* 
          Add the memory requirment of the DPU task -> DMA CMX2DDR direction (last DPU task) by adding a "MemoryRequirment" attribute to the flow in the MCM task graph.
          The memory requirment is defined as the output tensor (N*W*H*C) * dataType.
          This is required for the max topological cut pass.

          DMA/DPU -> Dealloc
        */
        int memoryRequirement = 1;
        auto dType = flowIt.source()->getInputTensor()[0]->get<mv::DType>("dType").getSizeInBits();
             
        for (unsigned int i = 0; i < flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
            memoryRequirement = flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

        memoryRequirement = memoryRequirement * dType/8;
        flowIt->set<int>("MemoryRequirement", memoryRequirement);
        flowIt->set<bool>("PositiveMemory", true); /*Required for transitive reduction filter*/

        
        for(auto sibling = parentOp.leftmostChild(); sibling != om.opEnd(); ++sibling)
        {
            if(sibling->getOpType() == "Deallocate")
                continue;

            // In flows
            if(!cm.checkControlFlow(sibling, op))
                cm.defineFlow(sibling, op);

            // Out flows
            for(auto dataNiece = sibling.leftmostChild(); dataNiece != om.opEnd(); ++dataNiece)
                if(dataNiece->getOpType() != "Deallocate")
                    if(!cm.checkControlFlow(op, dataNiece))
                        cm.defineFlow(op, dataNiece);

            for(auto controlNiece = cm.switchContext(sibling).leftmostChild(); controlNiece != cm.opEnd(); ++controlNiece)
                if(controlNiece->getOpType() != "Deallocate")
                    if(!cm.checkControlFlow(cm.switchContext(op), controlNiece))
                        cm.defineFlow(cm.switchContext(op), controlNiece);
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


