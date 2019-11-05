#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <unordered_set>

static void taskControlFlowsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void hangingDmaControlFlowsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::Element&);
static void NNCMX2DDRControlFlowsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void layerNumberingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);
static void activationTensorsControlFlowsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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

        MV_REGISTER_PASS(NNCMX2DDRControlFlows)
        .setFunc(NNCMX2DDRControlFlowsFcn)
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

// This pass is absolutely necessary to ensure that we are not redeaming in cmx weights too soon
// It is a conservative approach but it's needed for TIG
void NNCMX2DDRControlFlowsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto dmas = om.getOps("DMATask");

    std::vector<std::pair<mv::Control::OpListIterator, mv::Control::OpListIterator>> flowsToAdd;

    for(auto& dma : dmas)
    {
        auto direction = dma->get<mv::DmaDirection>("direction");
        if(direction == mv::NNCMX2DDR)
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
void hangingDmaControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
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
        // Check if it's DDR2NNCMX, otherwise we don't need to do anything
        if(dma->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2NNCMX)
        {
            // At this point (see assumption above) each DMA has at least one output control flow
            // We take the minimum layer number required by operations using this data

            // HACK: Horrible hack because apparentely we have problem in assigning iterators
            int sonWithMinimumLayerInvolvedIndex = 0;
            int minimumLayerInvolved = dma.leftmostChild()->get<unsigned>("layerNumber");

            unsigned i = 0;
            for(auto son = dma.leftmostChild(); son != cm.opEnd(); ++son)
            {
                int currentMinimumLayerInvolved = son->get<unsigned>("layerNumber");
                if(currentMinimumLayerInvolved < minimumLayerInvolved)
                {
                    minimumLayerInvolved = currentMinimumLayerInvolved;
                    sonWithMinimumLayerInvolvedIndex = i;
                }
                ++i;
            }

            auto sonWithMinimumLayerInvolved = dma.leftmostChild();
            for(int j = 0; j < sonWithMinimumLayerInvolvedIndex; ++j)
                ++sonWithMinimumLayerInvolved;

            // Now based on the prefetch we have to start from the sonWithMinimumLayerInvolved and go back prefetch layers
            for(auto positionInTopologicalSort = std::find(sortedOps.rbegin(), sortedOps.rend(), sonWithMinimumLayerInvolved); positionInTopologicalSort != sortedOps.rend(); ++positionInTopologicalSort)
            {
                auto preceedingOp = *positionInTopologicalSort;

                if (!preceedingOp->hasAttr("layerNumber") || preceedingOp->getOpType() == "DMATask")
                    continue;
                int preceedingOpLayerNumber = preceedingOp->get<unsigned>("layerNumber");

                // Two conditions must be true to build the control flow preceedingOp -> dma
                // 1) The difference in terms of layersNumber has to be greater or equal _dma_dependency
                // 2) There has to be a dependency between preceedingOp and the sonWithMinimumLayerInvolved (preeceding op could be on a parallel branch)
                if(minimumLayerInvolved - preceedingOpLayerNumber >= _dma_dependency &&
                   cm.pathExists(preceedingOp, sonWithMinimumLayerInvolved))
                {
                    flowsToAdd.push_back(std::make_pair(preceedingOp, dma));
                    break;
                }
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
// And possibly also to handle NNCMX2DDR output flows

// ASSUMPTION: We need task control flows and transitive reduction to be run before this pass
void layerNumberingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::ControlModel cm(model);

    unsigned initialLayerIndex = 0;
    std::unordered_set<std::string> firstIteration;
    firstIteration.insert(cm.getFirst()->getName());
    assignLayerNumber(cm, firstIteration, initialLayerIndex);
}

void addTaskControlFlowsAndRecursivelySkipImplicitOperationsDown(mv::OpModel& om, mv::Data::OpListIterator anchorOp, mv::Data::OpListIterator exploreOp)
{
    mv::ControlModel cm(om);

    for(auto nextOp = exploreOp.leftmostChild(); nextOp != om.opEnd(); ++nextOp)
    {
        if(!nextOp->hasTypeTrait("executable"))
            addTaskControlFlowsAndRecursivelySkipImplicitOperationsDown(om, anchorOp, nextOp);
        else if(cm.isFlowAllowedAndNonExisting(anchorOp, nextOp))
            cm.defineFlow(anchorOp, nextOp);
    }
}

void addTaskControlFlowsAndRecursivelySkipImplicitOperationsUp(mv::OpModel& om, mv::Data::OpListIterator anchorOp, mv::Data::OpListIterator exploreOp)
{
    mv::ControlModel cm(om);

    for(auto nextOp = exploreOp.leftmostParent(); nextOp != om.opEnd(); ++nextOp)
    {
        if(!nextOp->hasTypeTrait("executable"))
            addTaskControlFlowsAndRecursivelySkipImplicitOperationsUp(om, anchorOp, nextOp);
        else if(cm.isFlowAllowedAndNonExisting(nextOp, anchorOp))
            cm.defineFlow(nextOp, anchorOp);
    }
}


// This pass adds control flows relative to Task.
// Rationale: Each Task should be connected via a ControlFlow to the same operations he is connected via a DataFlow
// But implicit operations (e.g. Constants, Concat, Slice etc) must be skipped and/or avoided
void taskControlFlowsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto dmaTasks = om.getOps("DMATask");
    auto dpuTasks = om.getOps("DPUTask");
    auto upaTasks = om.getOps("UPATask");

    std::vector<mv::Data::OpListIterator> tasks;
    tasks.reserve(dmaTasks.size() + dpuTasks.size() + upaTasks.size());
    tasks.insert(tasks.end(), dmaTasks.begin(), dmaTasks.end());
    tasks.insert(tasks.end(), dpuTasks.begin(), dpuTasks.end());
    tasks.insert(tasks.end(), upaTasks.begin(), upaTasks.end());

    for(auto op : tasks)
    {
        addTaskControlFlowsAndRecursivelySkipImplicitOperationsDown(om, op, op);
        addTaskControlFlowsAndRecursivelySkipImplicitOperationsUp(om, op, op);
    }
}

void completeControlGraphFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::ControlModel cm(model);
    mv::OpModel om(model);
    auto output = cm.switchContext(om.getOutput());

    for(auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if(opIt.outputsSize() == 0)
            if(cm.isFlowAllowedAndNonExisting(opIt, output))
                cm.defineFlow(opIt, output);
    }
}

