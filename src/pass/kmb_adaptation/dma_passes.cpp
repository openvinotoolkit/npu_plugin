#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"

static void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddDPUTasksWeightsDMATasks)
            .setFunc(AddDPUTasksWeightsDMATasksFcn)
            .setDescription(
               "Add DMA Tasks for DPU Tasks weights");

        MV_REGISTER_PASS(AddUPATasksExtraInputsDMATasks)
            .setFunc(AddUPATasksExtraInputsDMATasksFcn)
            .setDescription(
               "Add DMA Tasks for UPA Tasks extra inputs");


    }
}

bool isTensorInNNCMX(mv::Data::TensorIterator tensor)
{
    if(tensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::NNCMX)
        return true;
    return false;
}

bool isTensorInUPACMX(mv::Data::TensorIterator tensor)
{
    if(tensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::UPACMX)
        return true;
    return false;
}

// Pass role: Add DMA Tasks for weights tensors input of DPUTasks (if needed).
void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element &)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto dpuTasks = om.getOps("DPUTask");

    for(auto& opIt : dpuTasks)
    {
        auto opId = opIt->get<unsigned>("opId");
        unsigned n = opIt->inputSlots();
        for(unsigned i = 1; i < n; ++i)
        {
            auto inputTensor = opIt->getInputTensor(i);
            mv::QuantizationParams quantParams = {{},{},{},{}};
            if(inputTensor->hasAttr("quantParams"))
                quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
            auto inputOp = om.getSourceOp(inputTensor);
            if(!isTensorInNNCMX(inputTensor))
            {
                auto flows = inputTensor->get<std::set<std::string>>("flows");
                mv::Data::TensorIterator inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2NNCMX, mv::createDMATaskDDR2NNCMXName(inputOp->getName()));
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

// Pass role: Add DMA Tasks for input tensors input of UPATasks (if needed).
void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element &)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto upaTasks = om.getOps("UPATask");

    for(auto& opIt : upaTasks)
    {
        std::string taskOp = opIt->get<std::string>("taskOp");
        if(taskOp == "Dummy")
            continue;
        auto opId = opIt->get<unsigned>("opId");
        unsigned n = opIt->inputSlots();
        for(unsigned i = 0; i < n; ++i)
        {
            auto inputTensor = opIt->getInputTensor(i);
            auto inputOp = om.getSourceOp(inputTensor);
            if(isTensorInNNCMX(inputTensor) || isTensorInUPACMX(inputTensor))
            {
                auto flows = inputTensor->get<std::set<std::string>>("flows");

                mv::Data::TensorIterator inputTensorDma;
                if(isTensorInNNCMX(inputTensor))
                    inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::NNCMX2DDR, mv::createDMATaskNNCMX2DDRName(inputOp->getName()));
                else if (isTensorInUPACMX(inputTensor))
                    inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::UPACMX2DDR, mv::createDMATaskUPACMX2DDRName(inputOp->getName()));
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
