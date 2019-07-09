#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void taskControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&);
static void cmx2DDRControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(HangingDmaControlFlows)
        .setFunc(hangingDmaControlFlowsFcn)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(TaskControlFlows)
        .setFunc(taskControlFlowsFcn)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(cmx2DDRControlFlows)
        .setFunc(cmx2DDRControlFlowsFcn)
        .setDescription(
            ""
        );

    }
}

// NOTE: This pass makes sense only when prefetch == 0
void cmx2DDRControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto dmas = om.getOps("DMATask");

    std::vector<std::pair<mv::Control::OpListIterator, mv::Control::OpListIterator>> flowsToAdd;

    for(auto& dma : dmas)
    {
        if(dma->get<mv::DmaDirection>("direction") == mv::CMX2DDR)
        {
            auto controlDma = cm.switchContext(dma);
            for(auto parent = controlDma.leftmostParent(); parent != cm.opEnd(); ++parent)
                for(auto sibling = parent.leftmostChild(); sibling != cm.opEnd(); ++sibling)
                    flowsToAdd.push_back(std::make_pair(controlDma, sibling));
        }
    }

    for(auto& flow : flowsToAdd)
        if(cm.isFlowAllowedAndNonExisting(flow.first, flow.second))
            cm.defineFlow(flow.first, flow.second);    
}

// This pass handles all the hanging DMAs into the graph using the prefetch logic
void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    int _dma_dependency = 0;
    if(passDesc.hasAttr("weights_prefetch"))
        _dma_dependency = passDesc.get<int>("weights_prefetch");

    auto sortedOps = cm.topologicalSort();

    auto dmas = om.getOps("DMATask");

    std::vector<std::pair<mv::Control::OpListIterator, mv::Control::OpListIterator>> flowsToAdd;

    if(_dma_dependency == 0)
    {
        // Simple strategy for _dma_dependency == 0
        // Check all the siblings of the hanging dma
        // If one has a parent, attach to the same parent
        // attach to the parent or to the sibling itself

        // As a general rule, we don't want to attach hanging dmas to other dmas

        // There is always a sibling with at least one parent,
        // this is ensured by the previous passes (DmaControlFlows and DpuControlFlows)

        // Problem with this approach: Let's say we are trying to attach control flows
        // for a dma involved into a dpu task in parallel branch:
        // The current approach attachs the control flow to the sibling, thus forbidding
        // real parallelism. How to solve this?
        for(auto dma : dmas)
        {
            auto dmaControl = cm.switchContext(dma);
            if(dmaControl.inputsSize() == 0)
            {
                //Collect siblings, nodes that share our same parent
                std::vector<mv::Control::OpListIterator> siblings;
                for(auto son = dmaControl.leftmostChild(); son != cm.opEnd(); ++son)
                    for(auto sibling = son.leftmostParent(); sibling != cm.opEnd(); ++ sibling)
                        siblings.push_back(sibling);

                bool allSiblingsAreDMA = true;
                std::vector<mv::Control::OpListIterator> dmaSiblings;
                for(auto& sibling : siblings)
                {
                    if(sibling.inputsSize() > 0)
                    {
                        if(sibling->getOpType() != "DMATask")
                        {
                            flowsToAdd.push_back(std::make_pair(sibling, dmaControl));
                            allSiblingsAreDMA = false;
                        }
                        else
                            dmaSiblings.push_back(sibling);
                    }
                }

                if(allSiblingsAreDMA)
                {
                    for(auto& dmaSibling : dmaSiblings)
                    {
                        for(auto positionInTopologicalSort = std::find(sortedOps.rbegin(), sortedOps.rend(), dmaSibling); positionInTopologicalSort != sortedOps.rend(); ++positionInTopologicalSort)
                        {
                            auto preceedingOp = *positionInTopologicalSort;
                            if(preceedingOp->getOpType() != "DMATask")
                            {
                                flowsToAdd.push_back(std::make_pair(preceedingOp, dmaControl));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Super aggressive prefetching: attach everything to Input:
    // Rationale: maxcut + partial serialization will solve all the problems of the world

    // This part is done in any case, regardless of the prefetching chosen, since transitive reduction will eventually eliminate the
    // extra control flows
    for(auto dma : dmas)
    {
        auto dmaControl = cm.switchContext(dma);
        if(dmaControl.inputsSize() == 0)
            flowsToAdd.push_back(std::make_pair(cm.getFirst(), dmaControl));
    }

    //TODO:Implement hybrid strategies

    for(auto& flowToAdd : flowsToAdd)
        if(cm.isFlowAllowedAndNonExisting(flowToAdd.first, flowToAdd.second))
            cm.defineFlow(flowToAdd.first, flowToAdd.second);
}

// This pass adds control flows relative to Task.
// Rationale: Each DMA Task should be connected via a ControlFlow to the same operations he is connected via a DataFlow
// But implicit operations (e.g. Constants, Concat, Slice etc) must be skipped and/or avoided

// NOTE: For now, only one level of implicit operations is handled. In the future we will need a recursive procedure
void taskControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto dmaTasks = om.getOps("DMATask");
    auto dpuTasks = om.getOps("DPUTask");

    std::vector<mv::Data::OpListIterator> tasks;
    tasks.reserve(dmaTasks.size() + dpuTasks.size());
    tasks.insert(tasks.end(), dmaTasks.begin(), dmaTasks.end());
    tasks.insert(tasks.end(), dpuTasks.begin(), dpuTasks.end());

    for(auto op : tasks)
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
