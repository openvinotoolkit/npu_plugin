#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void addDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void removeDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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
void addDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto dataFlowIt = dm.flowBegin(); dataFlowIt != dm.flowEnd(); ++dataFlowIt)
    {
        auto inputOp = dataFlowIt.source();
        auto outputOp = dataFlowIt.sink();

        if(outputOp->getOpType() == "Output")
            continue;

        if(inputOp->getOpType() == "Input")
            continue;

        if(inputOp->getOpType() == "Constant" || inputOp->getOpType() == "ConstantInt" || inputOp->getOpType() == "ConstantDataElement")
            continue;

        auto opId = inputOp->get<unsigned>("opId");
        auto inputOpName = inputOp->getName();
        auto inputTensor = dataFlowIt->getTensor();

        std::string deallocationName("Deallocate"+inputOpName);

        if(!om.checkOp(deallocationName))
            om.deallocate(inputTensor, deallocationName);
        auto deallocateInputOp = om.getOp(deallocationName);
        deallocateInputOp->set<unsigned>("opId", opId);
    }
}

// Pass role: Remove deallocation tasks for each Tensor

// Data flows should not be propagated, control flows yes
void removeDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);
    
    std::vector<std::pair<mv::Data::OpListIterator, mv::Data::OpListIterator>> newEdges ;

    for (auto ctlFlow = cm.getFirst(); ctlFlow != cm.getLast(); ++ctlFlow)
    {
        auto ctlFlowOpType = ctlFlow->getOpType();
        if (ctlFlowOpType == "Deallocate")
        {
            for ( auto parentOp = ctlFlow.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
            {
                for ( auto childOp = ctlFlow.leftmostChild(); childOp != cm.opEnd(); ++childOp)
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
                        cm.defineFlow(om.switchContext(parentOp), om.switchContext(childOp));
                    }
                }
            }
        }
    }

    auto deallocTasks = om.getOps("Deallocate");

    for(auto vecIt = deallocTasks.begin(); vecIt != deallocTasks.end(); ++vecIt)
    {
        auto deallocTaskDataIt = *vecIt;

        om.removeOp(deallocTaskDataIt);
    }
}