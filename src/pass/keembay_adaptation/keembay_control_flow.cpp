#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void taskControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&);
static void cmx2DDRControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&);
static void layerNumberingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::json::Object&);

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

        MV_REGISTER_PASS(LayerNumbering)
        .setFunc(layerNumberingFcn)
        .setDescription(
            ""
        );

    }
}

// NOTE: This pass makes sense only when hanging dmas have been solved
// and assign layer number has been rerun
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
// Weights prefetch controls how many levels before we want to start to load weights
// Minimum (and most conservative approach) is 1

// ASSUMPTION: This pass happens after the pass that assigns a layer number to each layer already in the control model
void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto sortedOps = cm.topologicalSort();

    int _dma_dependency = 1;
    if(passDesc.hasAttr("weights_prefetch"))
        _dma_dependency = passDesc.get<int>("weights_prefetch");

    auto dmas = om.getOps("DMATask");

    std::vector<std::pair<mv::Control::OpListIterator, mv::Control::OpListIterator>> flowsToAdd;

    for(auto& dmaOp: dmas)
    {
        auto dma = cm.switchContext(dmaOp);
        // Check if it's hanging, otherwise we don't need to do anything
        if(dma.inputsSize() == 0)
        {
            // At this point (see assumption above) each DMA has at least one output control flow
            // We take the minimum layer number required by operations using this data

            // HACK: Horrible hack because apparentely we have problem in assigning iterators
            unsigned sonWithMinimumLayerInvolvedIndex = 0;
            unsigned minimumLayerInvolved = dma.leftmostChild()->get<unsigned>("layerNumber");

            unsigned i = 0;
            for(auto son = dma.leftmostChild(); son != cm.opEnd(); ++son)
            {
                unsigned currentMinimumLayerInvolved = son->get<unsigned>("layerNumber");
                if(currentMinimumLayerInvolved < minimumLayerInvolved)
                {
                    minimumLayerInvolved = currentMinimumLayerInvolved;
                    sonWithMinimumLayerInvolvedIndex = i;
                }
                ++i;
            }

            auto sonWithMinimumLayerInvolved = dma.leftmostChild();
            for(unsigned j = 0; j < sonWithMinimumLayerInvolvedIndex; ++j)
                ++sonWithMinimumLayerInvolved;

            // Now based on the prefetch we have to start from the sonWithMinimumLayerInvolved and go back prefetch layers
            for(auto positionInTopologicalSort = std::find(sortedOps.rbegin(), sortedOps.rend(), sonWithMinimumLayerInvolved); positionInTopologicalSort != sortedOps.rend(); ++positionInTopologicalSort)
            {
                auto preceedingOp = *positionInTopologicalSort;

                if (!preceedingOp->hasAttr("layerNumber"))
                    continue;
                unsigned preceedingOpLayerNumber = preceedingOp->get<unsigned>("layerNumber");

                // Two conditions must be true to build the control flow preceedingOp -> dma
                // 1) The difference in terms of layersNumber has to be EXACTLY _dma_dependency
                // 2) There has to be a dependency between preceedingOp and the sonWithMinimumLayerInvolved (preeceding op could be on a parallel branch)
                if(minimumLayerInvolved - preceedingOpLayerNumber == _dma_dependency && cm.pathExists(preceedingOp, sonWithMinimumLayerInvolved))
                    flowsToAdd.push_back(std::make_pair(preceedingOp, dma));

            }
        }
    }

    for(auto& flowToAdd : flowsToAdd)
        if(cm.isFlowAllowedAndNonExisting(flowToAdd.first, flowToAdd.second))
            cm.defineFlow(flowToAdd.first, flowToAdd.second);
}

void assignLayerNumber(mv::ControlModel& cm, const std::unordered_set<std::string>& opNames, unsigned indexToAssign)
{
    if(opNames.empty())
        return;

    std::unordered_set<std::string> nextIteration;
    for(auto& opName: opNames)
    {
        auto op = cm.switchContext(cm.getOp(opName));
        op->set<unsigned>("layerNumber", indexToAssign);
        for(auto son = op.leftmostChild(); son != cm.opEnd(); ++son)
            nextIteration.insert(son->getName());
    }

    assignLayerNumber(cm, nextIteration, ++indexToAssign);
}


// This pass adds a numeric index that stands for layer to each op
// It will be useful to solve hanging DMA's with a proper prefetch routine
// And possibly also to handle CMX2DDR output flows

// ASSUMPTION: We need task control flows and transitive reduction to be run before this pass
void layerNumberingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::ControlModel cm(model);

    unsigned initialLayerIndex = 0;
    std::unordered_set<std::string> firstIteration;
    firstIteration.insert(cm.getFirst()->getName());
    assignLayerNumber(cm, firstIteration, initialLayerIndex);
}

// This pass adds control flows relative to Task.
// Rationale: Each DMA Task should be connected via a ControlFlow to the same operations he is connected via a DataFlow
// But implicit operations (e.g. Constants, Concat, Slice etc) must be skipped and/or avoided

// NOTE: For now, only max two level of implicit operations is handled. In the future we will need a recursive procedure
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
        for(auto son = op.leftmostChild(); son != om.opEnd(); ++son)
        {
            if(!son->hasTypeTrait("executable"))
            {
                for (auto nephew = son.leftmostChild(); nephew != om.opEnd(); ++nephew)
                {
                    // This hack is horrible and should be substituted with a recursive procedure ASAP
                    if(!nephew->hasTypeTrait("executable"))
                    {
                        for(auto protoNephew = nephew.leftmostChild(); protoNephew != om.opEnd(); ++protoNephew)
                        {
                            if(cm.isFlowAllowedAndNonExisting(op, protoNephew))
                                cm.defineFlow(op, protoNephew);
                        }
                    }
                    else if(cm.isFlowAllowedAndNonExisting(op, nephew)) 
                        cm.defineFlow(op, nephew);
                }
            }
            else if(cm.isFlowAllowedAndNonExisting(op, son))
                cm.defineFlow(op, son);
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
