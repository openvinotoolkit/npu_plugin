#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void taskControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&);

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

    }
}


// This pass handles all the hanging DMAs into the graph using the prefetch logic
void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    //dma_dependency = std::min(std::max(1, self.nn_cmx_memory/param.cluster_size), dma_dependency);
    //    This is the weights prefetch number. It specifies how early to start the inbound DMA for weights.
    //    The units are number of ops preceeding current conv in the topographically sorted ops list.
    //    If the weights tensor is very large (eg > 1/2 of CMX) then the specified prefetch parameter (eg 2)
    //    would be reduced. This assumes that the only fit partial serialization would find for such a
    //    big weights tensor would be to start the DMA right before it is needed. For smaller tensors, the
    //    user-specified prefetch number will be used. The prefetch edge added is for partial serialization.
    //

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    auto cmxSize = globalConfigParams->get<unsigned>("cmx");

    int _dma_dependency = 0;
    if(passDesc.hasAttr("weights_prefetch"))
        _dma_dependency = passDesc.get<int>("weights_prefetch");
    int dma_dependency;

    auto removeOpsBasedOnOpType = [] (std::vector<mv::Control::OpListIterator>& list, const std::string& opType)
    {
        list.erase(std::remove_if(list.begin(), list.end(), [opType](mv::Control::OpListIterator it) { return it->getOpType() == opType;}), list.end());
    };

    auto sortedOps = cm.topologicalSort();

    // Simple strategy for _dma_dependency == 0
    // Check all the siblings of the hanging dma
    // If one has a parent, attach to the same parent
    // attach to the parent or to the sibling itself

    // As a general rule, we don't want to attach hanging dmas to other dmas

    // There is always a sibling with at least one parent,
    // this is ensured by the previous passes (DmaControlFlows and DpuControlFlows)

    std::vector<std::pair<mv::Control::OpListIterator, mv::Control::OpListIterator>> flowsToAdd;

    if(_dma_dependency == 0)
    {
        auto dmas = om.getOps("DMATask");
        for(auto dma : dmas)
        {
            auto dmaControl = cm.switchContext(dma);
            if(dmaControl.inputsSize() == 0)
            {
                //Collect siblings
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

        for(auto& flowToAdd : flowsToAdd)
            if(cm.isFlowAllowedAndNonExisting(flowToAdd.first, flowToAdd.second))
                cm.defineFlow(flowToAdd.first, flowToAdd.second);

    }

    //NOTE: This will change with multicluster
//    long unsigned inputTensorDmaDimension = inputTensorDma->computeTotalSize();
//    for(unsigned j = 0; j < inputOutputTensors; ++j)
//        inputTensorDmaDimension += opIt->getInputTensor(j)->computeTotalSize();

//    int partsPerCMX = std::max((unsigned long)1, cmxSize/inputTensorDmaDimension);
//    if (partsPerCMX < (_dma_dependency + 1))
//        dma_dependency = partsPerCMX;
//    else
//        dma_dependency =  _dma_dependency + 1 ;


//    auto index = std::distance(sortedOps.begin(), std::find(sortedOps.begin(), sortedOps.end(), opIt));
//    if(index <= dma_dependency)
//        cm.defineFlow(om.getInput(), inputTensorDmaOp);
//    else
//    {
//        std::cout << "NOT ATTACHING TO INPUT" << std::endl;
//        cm.defineFlow(sortedOps[index - dma_dependency], inputTensorDmaOp);
//    }
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
