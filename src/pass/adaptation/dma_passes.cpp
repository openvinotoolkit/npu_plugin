#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void addDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void addFinalDMATaskFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);


namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddDMATasks)
            .setFunc(addDMATasksFcn)
            .setDescription(
               "Add DMA Tasks where needed in the Task graph");

        MV_REGISTER_PASS(AddFinalDMATask)
            .setFunc(addFinalDMATaskFcn)
            .setDescription(
               "Add final DMA task for output");
    }
}

// ASSUMPTION: If a tensor comes from a DDR2CMX dMATask or a Task in general, then it's already in CMX
// and does not need to be transfered. In all other cases, it needs to be transfered.

// NOTE: This is not checked using allocators for the simple reason that they are not assigned
// to tensors yet.
bool isTensorInCMX(mv::Data::TensorIterator tensor, mv::BaseOpModel& opModel)
{
    auto sourceOp = opModel.getSourceOp(tensor);
    std::string opType(sourceOp->getOpType());
    if(opType == "DMATask")
    {
        if(sourceOp->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2CMX)
            return true;
        else
            return false;
    }
    else if(opType.find("Task") != std::string::npos)
        return true;
    else
        return false;
}

void addFinalDMATaskFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto opIt = om.getOutput();
    auto input = opIt->getInputTensor(0);

    auto opId = opIt->get<unsigned>("opId");
    std::string oldOutputName(opIt->getName());

    if(isTensorInCMX(input, om))
    {
        auto newInput = om.dMATask(input, mv::DmaDirectionEnum::CMX2DDR);
        om.getSourceOp(newInput)->set<unsigned>("opId", opId);
        auto backup = opIt;
        om.removeOp(backup);
        om.output(newInput, oldOutputName);
        om.getOp(oldOutputName)->set<unsigned>("opId", opId);
    }
}

void addDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // Pass main assumption is that we are working on the original graph, just with the Ops converted to DPUTasks
    // We don't need to perform eliminations in this pass, we can use a for loop to iterate among operations
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "DPUTask")
        {
            auto opId = opIt->get<unsigned>("opId");
            auto flow = opIt.leftmostInput();
            for(unsigned i = 0; i < opIt->inputSlots(); ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                auto inputOp = om.getSourceOp(inputTensor);
                auto inputOpName = inputOp->getName();
                std::string inputDeallocationName("Deallocate"+inputOpName);
                if(!isTensorInCMX(inputTensor, om))
                {
                    auto inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2CMX);
                    auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                    inputTensorDmaOp->set<unsigned>("opId", opId);

                    auto backupFlow = flow;
                    ++flow;
                    om.undefineFlow(backupFlow);
                    opIt->setInputTensor(inputTensorDma, i, false);
                    om.defineFlow(inputTensorDmaOp, 0, opIt, i);
                    inputTensor = inputTensorDma;
                }
                else
                    ++flow;

                if(!om.checkOp(inputDeallocationName))
                    om.deallocate(inputTensor, inputDeallocationName);
                auto deallocateInputOp = om.getOp(inputDeallocationName);
                deallocateInputOp->set<unsigned>("opId", opId);
                cm.defineFlow(opIt, deallocateInputOp);
            }
        }
    }
}
