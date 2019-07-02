#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include <algorithm>

static void addDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void removeDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

// Pass role: Add deallocation tasks for each Tensor
void addDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

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
            inputOp->getOpType() == "WeightsTable" || inputOp->getOpType() == "SparsityMap")
            continue;

        auto inputTensor = dataFlowIt->getTensor();
        
        if (inputTensor->get<mv::Tensor::MemoryLocation>("Location") != mv::Tensor::MemoryLocation::CMX)
            continue;
        
        // Tensors that are input of a concat shall not be deallocated: they will be allocated into a bigger tensor
        // (the output of concat op) and that will be deallocated

        // ADDITIONAL NOTE/TODO: This check by itself is not sufficient if a tensor is input of both an implicit and an explicit operation
        if(!outputOp->hasTypeTrait("executable"))
            continue;

        auto inputOpName = inputOp->getName();

        std::string deallocationName(mv::createDeallocationName(inputOpName));

        // Each tensor must be deallocated once
        if(!om.checkOp(deallocationName))
        {
            // Flows names must be taken before the insertion of deallocation ops
            // Otherwise deallocation will appear as well
            auto flowsNames = inputTensor->get<std::set<std::string>>("flows");

            mv::Data::OpListIterator deallocateInputOp;

            if(outputOp->getOpType() == "DMATask" && outputOp->get<mv::DmaDirection>("direction") == mv::CMX2DDR)
                deallocateInputOp = outputOp;
            else
            {
                // Creating deallocation operation for the tensor and attaching it through a dataflow
                // to the operation that created it
                om.deallocate(inputTensor, deallocationName);
                deallocateInputOp = om.getOp(deallocationName);
            }

            /* Concat memory requirment should be 0 */
            if(dataFlowIt.source()->getOpType() != "ImplicitConcat") 
            {
                // Attaching also through a ControlFlow
                if(cm.isFlowAllowed(inputOp, deallocateInputOp))
                {
                    mv::Control::FlowListIterator flowIt = cm.checkControlFlow(inputOp, deallocateInputOp);
                    if(flowIt == cm.flowEnd())
                        flowIt = cm.defineFlow(inputOp, deallocateInputOp);
                    auto outputTensor = flowIt.source()->getOutputTensor(0);
                    flowIt->set<int>("MemoryRequirement", outputTensor->computeTotalSize());
                    flowIt->set<bool>("PositiveMemory", true);
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
                        flowIt->set<int>("MemoryRequirement", outputTensor->computeTotalSize());
                        flowIt->set<bool>("PositiveMemory", true);
                    }
                }
            }
            

            // Control flows for the newly created operation must be attached now.
            // Checking all the ops that have this tensor as input

            std::vector<mv::Data::OpListIterator> sinkOperations;
            for(auto flowName : flowsNames)
            {
                auto df = dm.getDataFlow(flowName);
                sinkOperations.push_back(df.sink());
            }

            bool found = false;
            auto chosenOp = sortedOps.rbegin();
            for(; chosenOp != sortedOps.rend(); ++chosenOp)
            {
                if(std::find(sinkOperations.begin(), sinkOperations.end(), om.switchContext(*chosenOp)) != sinkOperations.end())
                {
                    found = true;
                    break;
                }
            }

            if(!found)
                throw mv::RuntimeError(cm, "Something is wrong in deallocation pass");


            if(cm.isFlowAllowedAndNonExisting(om.switchContext(*chosenOp), deallocateInputOp))
                cm.defineFlow(om.switchContext(*chosenOp), deallocateInputOp);

            // This loop has to happen in both data and control model
            // DATA
            auto chosenOpData = om.switchContext(*chosenOp);
            for(auto son = chosenOpData.leftmostChild(); son != om.opEnd(); ++son)
            {
                if(!son->hasTypeTrait("executable") && son->getOpType() != "Output")
                {
                    // Concat is brutally skipped - No need to check in control model
                    // since there will NEVER BE a control flow connected to a concat
                    // NOTE/TODO: We have to go down recursively for the concat of concats case
                    // For now we stop at the first level.

                    for(auto nephew = son.leftmostChild(); nephew != om.opEnd(); ++nephew)
                        if(cm.isFlowAllowedAndNonExisting(deallocateInputOp, nephew))
                            cm.defineFlow(deallocateInputOp, nephew);
                }

                else if(cm.isFlowAllowedAndNonExisting(deallocateInputOp, son))
                {
                    if(son->getOpType() != "Deallocate")
                        cm.defineFlow(deallocateInputOp, son);

                }
            }

            // CONTROL
            auto chosenOpControl = *chosenOp;
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
    }
}

// Pass role: Remove deallocation tasks for each Tensor

// Data flows should not be propagated, control flows yes
void removeDeallocationTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
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
