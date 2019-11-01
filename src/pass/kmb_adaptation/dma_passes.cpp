#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"

static void addWeightsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);
static void addFinalDMATaskFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddWeightsDMATasks)
            .setFunc(addWeightsDMATasksFcn)
            .setDescription(
               "Add Weights DMA Tasks where needed in the Task graph");

        MV_REGISTER_PASS(AddFinalDMATask)
            .setFunc(addFinalDMATaskFcn)
            .setDescription(
               "Add initial and final DMA task in the Task graph");

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
    else if(opType == "ConstantInt" || opType == "Constant" || opType == "ConstantDataElement")
        return false;
    else if(opType == "Input")
        return false;
    else
        return true;
}

// Pass role: Add initial and final DMA Task CMX2DDR (if needed)
void addFinalDMATaskFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // OUTPUT
    auto opIt = om.getOutput();
    auto input = opIt->getInputTensor(0);
    auto inputOp = om.getSourceOp(input);

    auto opId = opIt->get<unsigned>("opId");
    std::string oldOutputName(opIt->getName());
    mv::QuantizationParams quantParams = {{},{},{},{}};
    if(input->hasAttr("quantParams"))
        quantParams = input->get<mv::QuantizationParams>("quantParams");
    if(isTensorInCMX(input, om))
    {
        auto newInput = om.dMATask(input, mv::DmaDirectionEnum::CMX2DDR, mv::createDMATaskCMX2DDRName(inputOp->getName()));
        auto newInputOp = om.getSourceOp(newInput);
        newInputOp->set<unsigned>("opId", opId);
        auto backup = opIt;
        om.removeOp(backup);
        om.output(newInput, quantParams, oldOutputName);
        auto newOutputOp = om.getOp(oldOutputName);
        newOutputOp->set<unsigned>("opId", opId);
    }
}


// Pass role: Add DMA Task DDR2CMX where needed for weights tensors input of DPUTasks.
void addWeightsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // Pass main assumption is that we are working on the original graph, just with the Ops converted to DPUTasks
    // We don't need to perform eliminations in this pass, we can use a for loop to iterate among operations
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "DPUTask")
        {
            auto opId = opIt->get<unsigned>("opId");
            unsigned n = opIt->inputSlots();
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                mv::QuantizationParams quantParams = {{},{},{},{}};
                if(inputTensor->hasAttr("quantParams"))
                    quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
                auto inputOp = om.getSourceOp(inputTensor);
                if(!isTensorInCMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");


                    auto inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2CMX, mv::createDMATaskDDR2CMXName(inputOp->getName()));
                    auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                    inputTensorDmaOp->set<unsigned>("opId", opId);

                    for(auto flowStr: flows)
                    {
                        auto backupFlow = dm.getDataFlow(flowStr);
                        auto idx = backupFlow->get<std::size_t>("sinkInput");
                        auto sink = backupFlow.sink();
                        om.undefineFlow(backupFlow);
                        sink->setInputTensor(inputTensorDma, idx, false);
                        om.defineFlow(inputTensorDmaOp, 0, sink, idx);
                    }
                }
            }
        }
    }
}
