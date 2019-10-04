#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include <algorithm>

static void addDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::Element&);
static void removeDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddDeallocationTasks)
            .setFunc(addDeallocationTasksFcn)
            .setDescription(
               "Add deallocation tasks where needed in the Task graph");

        MV_REGISTER_PASS(RemoveDeallocationTasks)
            .setFunc(removeDeallocationTasksFcn)
            .setDescription(
               "Remove all deallocation tasks from the Task graph");
    }
}

void insertDeallocationControlFlows(mv::OpModel& om, mv::Data::OpListIterator deallocateInputOp, mv::Control::OpListIterator chosenOp)
{
    mv::ControlModel cm(om);

    // Deallocation in flow
    if(cm.isFlowAllowedAndNonExisting(om.switchContext(chosenOp), deallocateInputOp))
        cm.defineFlow(om.switchContext(chosenOp), deallocateInputOp);

    // Deallocation out flows
    auto chosenOpControl = chosenOp;
    auto deallocateInputOpControl = cm.switchContext(deallocateInputOp);
    for(auto son = chosenOpControl.leftmostChild(); son != cm.opEnd(); ++son)
    {
        if(cm.isFlowAllowedAndNonExisting(deallocateInputOpControl, son))
        {
            if(son->getOpType() != "Deallocate")
                cm.defineFlow(deallocateInputOpControl, son);
        }
    }
}

bool thereIsDependency(mv::ControlModel& cm, const std::vector<mv::Data::OpListIterator>& sinkOperations)
{
    // To check if there is dependency, we have to check the existence of a control flow path between each pair
    // of operations
    unsigned n = sinkOperations.size();
    for(unsigned i = 0; i < n - 1; ++i)
        for(unsigned j = i + 1; j < n; ++j)
            if(cm.pathExists(cm.switchContext(sinkOperations[i]), cm.switchContext(sinkOperations[j])))
                return true;
    return false;
}


// Pass role: Add deallocation tasks for each Tensor
void addDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    // Sorting ops in dataflow topological order. Will be needed later.
    auto sortedOps = cm.topologicalSort();

    for(auto dataFlowIt = dm.flowBegin(); dataFlowIt != dm.flowEnd(); ++dataFlowIt)
    {
        auto inputOp = dataFlowIt.source();
        auto outputOp = dataFlowIt.sink();

        // Final tensor shall not be deallocated
        if(outputOp->getOpType() == "Output")
            continue;

        // Input tensor shall not be deallocated (will be DMAed, then that will be deallocated)
        if(inputOp->getOpType() == "Input")
            continue;

        // Constant tensors shall not be deallocated (will be DMAed, then those will be deallocated)
        // NOTE: This check must remain as is, can't be replaced by hasTypeTrait("executable") check
        // e.g. concat's output tensor needs deallocation even if it's not executable
        if(inputOp->getOpType() == "Constant" || inputOp->getOpType() == "ConstantInt" || inputOp->getOpType() == "ConstantDataElement" ||
           inputOp->getOpType() == "Slice")
            continue;

        auto inputTensor = dataFlowIt->getTensor();
        //We just have to check if it was previously deallocated or not.

        if(!inputTensor->hasAttr("deallocated") &&
          ((inputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::CMX &&
                outputOp->hasTypeTrait("executable")) ||
          (inputOp->getOpType() == "DMATask" && inputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::DDR)))
        {

            auto opType = inputOp->getOpType();
            inputTensor->set<bool>("deallocated", true);
            auto inputOpName = inputOp->getName();

            std::string deallocationName(mv::createDeallocationName(inputOpName));

            // Flows names must be taken before the insertion of deallocation ops
            // Otherwise deallocation will appear as well
            auto flowsNames = inputTensor->get<std::set<std::string>>("flows");

            mv::Data::OpListIterator deallocateInputOp;

            om.deallocate(inputTensor, deallocationName);
            deallocateInputOp = om.getOp(deallocationName);
            deallocateInputOp->set<mv::Tensor::MemoryLocation>("Location", inputTensor->get<mv::Tensor::MemoryLocation>("Location"));

            // Now that we have created/set the pointer to/ the deallocation op, we have to attach
            // the control flows to it such that it respects the properties of a dataflow graph
            // described here:

            //https://ieeexplore.ieee.org/document/8425174

            // We start with the control flow that carries the memory requirement.
            // For all cases but Implicit Concat, the rule is pretty simple: There is one control flow
            // coincident with the data flow that carries the memory requirement
            if(inputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::CMX)
            {
                if(inputOp->getOpType() != "ImplicitConcat")
                {
                    if(cm.isFlowAllowed(inputOp, deallocateInputOp))
                    {
                        // Check if the flow already exists, otherwise creating it
                        mv::Control::FlowListIterator flowIt = cm.checkControlFlow(inputOp, deallocateInputOp);
                        if(flowIt == cm.flowEnd())
                            flowIt = cm.defineFlow(inputOp, deallocateInputOp);
                        auto outputTensor = flowIt.source()->getOutputTensor(0);
                        flowIt->set<int>("MemoryRequirement", outputTensor->getClusterSize(16,false));
                    }
                }
                else
                {
                    // For concat, we don't need a single control flow going to the concat
                    // to the dealloc, but multiple control flows going from each of the concats
                    // inputs to the dealloc
                    auto concatOp = dataFlowIt.source();
                    for(auto concatInput = concatOp.leftmostParent(); concatInput != om.opEnd(); ++concatInput)
                    {
                        // Attaching also through a ControlFlow
                        if(cm.isFlowAllowed(concatInput, deallocateInputOp))
                        {
                            mv::Control::FlowListIterator flowIt = cm.checkControlFlow(concatInput, deallocateInputOp);
                            if(flowIt == cm.flowEnd())
                                flowIt = cm.defineFlow(concatInput, deallocateInputOp);
                            auto outputTensor = flowIt.source()->getOutputTensor(0);
                            flowIt->set<int>("MemoryRequirement", outputTensor->getClusterSize(16,true));
                        }
                    }
                }
            }

            std::vector<mv::Data::OpListIterator> sinkOperations;
            if (inputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::CMX &&
                outputOp->hasTypeTrait("executable"))
            {
                pass.log(mv::Logger::MessageType::Debug, " Collecting Sink Operations for case CMX Dealloc");

                // Now it's time to define the control flow with 0 as memory requirement
                // Which in our case is no memory requirement at all.

                // Checking all the ops that have this tensor as input

                for(auto flowName : flowsNames)
                {
                    pass.log(mv::Logger::MessageType::Debug, " Checking flow name " + flowName);

                    auto df = dm.getDataFlow(flowName);
                    sinkOperations.push_back(df.sink());
                }
            }
            else
            {
                pass.log(mv::Logger::MessageType::Debug, " Collecting Sink Operations for case DDR Dealloc");

                while(!flowsNames.empty())
                {
                    auto flowName = flowsNames.begin();
                    pass.log(mv::Logger::MessageType::Debug, " Checking flow name " + *flowName);

                    auto df = dm.getDataFlow(*flowName);
                    auto chosenOp = cm.switchContext(df.sink());
                    if (!chosenOp->hasTypeTrait("executable") || chosenOp->getOpType() == "DMATask")
                    {
                        //if it's a DMAtask it might be prefetched, so it's topological sort will not give us the right order
                        // so let's look for DPUTasks that this DMA goes to.
                        auto implicitOpFlowsNames = chosenOp->getOutputTensor(0)->get<std::set<std::string>>("flows");
                        flowsNames.insert(implicitOpFlowsNames.begin(), implicitOpFlowsNames.end());
                    }
                    else
                    {
                        if (chosenOp->getOpType() != "Deallocate") //really, not interested in Deallocate sinks
                           sinkOperations.push_back(df.sink());
                    }
                    flowsNames.erase(flowName);
                }
            }
            // Attaching to the last one in reverse topological order
            auto chosenOp = sortedOps.rbegin();
            for(; chosenOp != sortedOps.rend(); ++chosenOp)
                if(std::find(sinkOperations.begin(), sinkOperations.end(), om.switchContext(*chosenOp)) != sinkOperations.end())
                    break;
            insertDeallocationControlFlows(om, deallocateInputOp, *chosenOp);
        }
    }
}

// Pass role: Remove deallocation tasks for each Tensor

// Data flows should not be propagated, control flows yes
void removeDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    std::vector<std::pair<mv::Data::OpListIterator, mv::Data::OpListIterator>> oldEdges ;
    std::vector<std::pair<mv::Data::OpListIterator, mv::Data::OpListIterator>> newEdges ;

    // build a list of all the control flow edges
    for (auto ctlFlow = om.opBegin(); ctlFlow != om.opEnd(); ++ctlFlow)
    {
        auto cmCtlFlow = cm.switchContext(ctlFlow);
        for ( auto parentOp = cmCtlFlow.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
        {
            std::pair<mv::Data::OpListIterator, mv::Data::OpListIterator> oldEdge(om.switchContext(parentOp), ctlFlow);
            oldEdges.push_back(oldEdge);
        }
    }

    // build list of edges to add around deallocs
    for (auto ctlFlow = om.opBegin(); ctlFlow != om.opEnd(); ++ctlFlow)
    {
        auto ctlFlowOpType = ctlFlow->getOpType();
        if (ctlFlowOpType == "Deallocate")
        {
            auto cmCtlFlow = cm.switchContext(ctlFlow);
            for (auto parentOp = cmCtlFlow.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
            {
                // Implicit operations that go to dealloc shall not propagate control flow
                if(!parentOp->hasTypeTrait("executable"))
                    continue;

                for (auto childOp = cmCtlFlow.leftmostChild(); childOp != cm.opEnd(); ++childOp)
                {
                    auto childOpName = childOp->getName();

                    // add edge around dealloc if it does not exist
                    bool AddNewEdge = true ;
                    for ( auto tryEdge : newEdges )
                    {
                        if ((tryEdge.first->getName()==parentOp->getName()) && (tryEdge.second->getName()==childOp->getName()))
                        {
                            AddNewEdge = false;
                            break;
                        }
                    }
                    if (AddNewEdge)
                    {
                        std::pair<mv::Data::OpListIterator, mv::Data::OpListIterator> newEdge(om.switchContext(parentOp), om.switchContext(childOp));
                        newEdges.push_back(newEdge);
                    }
                }
            }
        }
    }

    // insert the new edges around the deallocs into the graph
    for (auto newPair : newEdges )
    {
        // do not add edge if it already was there
        bool AddNewEdge = true ;
        for ( auto tryEdge : oldEdges )
        {
            if ((tryEdge.first->getName()==newPair.first->getName()) && (tryEdge.second->getName()==newPair.second->getName()))
            {

                AddNewEdge = false;
                break;
            }
        }
        if (AddNewEdge)
        {
            cm.defineFlow(newPair.first , newPair.second );
        }
    }

    // remove the deallocs
    auto deallocTasks = om.getOps("Deallocate");
    for(auto vecIt = deallocTasks.begin(); vecIt != deallocTasks.end(); ++vecIt)
    {
        auto deallocTaskDataIt = *vecIt;
        om.removeOp(deallocTaskDataIt);
    }
}
