#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"

static void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void CleanRedundantDMAsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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

        MV_REGISTER_PASS(CleanRedundantDMAs)
            .setFunc(CleanRedundantDMAsFcn)
            .setDescription(
               "Remove DMAs that move data unnecessarily (e.g. DDR->NNCMX->DDR)");
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
        // Note: Marking pipelining hints for the scheduler, these DMAs can be overlapped with corresponding DPU tasks
        // These strategy choices come from the GO/SM
        bool pipelineInput = opIt->hasAttr("pipelining") && 
                                    (opIt->get<std::string>("pipelining") == "PipelineActivations");
        bool pipelineWeights = opIt->hasAttr("pipelining") && 
                                    (opIt->get<std::string>("pipelining") == "PipelineWeights");
        unsigned pipelineId = 0;
        if(opIt->hasAttr("schedule_for_dpu_dma_overlap"))
            pipelineId = opIt->get<unsigned>("schedule_for_dpu_dma_overlap");

        auto opId = opIt->get<unsigned>("opId");
        unsigned n = opIt->inputSlots();
        //Note: Adds dmas not only for weights like the name says...
        for(unsigned i = 0; i < n; ++i)
        {
            auto inputTensor = opIt->getInputTensor(i);
            auto inputOp = om.getSourceOp(inputTensor);
            if(!isTensorInNNCMX(inputTensor))
            {
                auto flows = inputTensor->get<std::set<std::string>>("flows");
                mv::Data::TensorIterator inputTensorDma = om.dMATask(mv::createDMATaskDDR2NNCMXName(inputOp->getName()), inputTensor, mv::DmaDirectionEnum::DDR2NNCMX, 0);
                if (opIt->hasAttr("slicedInput3DDMA") &&
                     opIt->get<bool>("slicedInput3DDMA") && !inputTensor->isPopulated())
                {
                    inputTensor->set<bool>("dilatedSlices3DDMA", true);
                    inputTensor->set<unsigned>("dilationFactor",
                                              opIt->get<unsigned>("originalDilationFactor"));
                    inputTensor->set<std::size_t>("lineofConcatHeight",
                                                opIt->get<std::vector<std::size_t>>("subConvsCoordinates")[0]);
                    inputTensor->set<std::size_t>("inputConcatTensorIdx",
                                                opIt->get<std::vector<std::size_t>>("subConvsCoordinates")[1]);
                    if (opIt->hasAttr("streamHId"))
                    {
                        auto streamHId = opIt->get<unsigned>("streamHId");
                        inputTensor->set<unsigned>("streamHId", streamHId);
                    }

                }
                inputTensorDma->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                inputTensorDmaOp->set<unsigned>("opId", opId);
                if(pipelineInput && inputOp->getOpType() == "Slice")
                    inputTensorDmaOp->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
                if(pipelineWeights && inputOp->getOpType() != "Slice")
                    inputTensorDmaOp->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
                if(i==1 && opIt->hasAttr("multiple_weight_out_degree"))
                    inputTensorDmaOp->set<bool>("multiple_weight_out_degree", true);

                for(auto flowStr: flows)
                {
                    auto backupFlow = dm.getDataFlow(flowStr);
                    auto idx = backupFlow->get<std::size_t>("sinkInput");
                    auto sink = backupFlow.sink();
                    if (sink->getOpType() == "DMATask")
                        if (sink->get<mv::DmaDirection>("direction") == mv::DDR2NNCMX)
                            continue;
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
                    inputTensorDma = om.dMATask(mv::createDMATaskNNCMX2DDRName(inputOp->getName()), inputTensor, mv::DmaDirectionEnum::NNCMX2DDR, 0);
                else if (isTensorInUPACMX(inputTensor))
                    inputTensorDma = om.dMATask(mv::createDMATaskUPACMX2DDRName(inputOp->getName()), inputTensor, mv::DmaDirectionEnum::UPACMX2DDR, 0);
                auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                inputTensorDmaOp->set<unsigned>("opId", opId);

                for(auto flowStr: flows)
                {
                    auto backupFlow = dm.getDataFlow(flowStr);
                    auto idx = backupFlow->get<std::size_t>("sinkInput");
                    auto sink = backupFlow.sink();
                    if (sink->getOpType() != "UPATask")
                        continue;
                    om.undefineFlow(backupFlow);
                    sink->setInputTensor(inputTensorDma, idx, false);
                    om.defineFlow(inputTensorDmaOp, 0, sink, idx);
                }
            }
        }
    }
}

void CleanRedundantDMAsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element &)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto isException = [](const mv::Data::OpListIterator& firstDma, const mv::Data::OpListIterator& secondDma) {
        return (firstDma->hasAttr("explicitRelocate") && firstDma->get<bool>("explicitRelocate")) ||
               (secondDma->hasAttr("explicitRelocate") && secondDma->get<bool>("explicitRelocate"));
    };

    std::vector<mv::Data::OpListIterator> dmasToRemove;

    auto dmaOps = om.getOps("DMATask");
    for (auto& dmaOp : dmaOps) {
        if (dmaOp->get<mv::DmaDirection>("direction") != mv::DmaDirectionEnum::NNCMX2DDR)
            continue;

        const auto inputTensor = dmaOp->getInputTensor(0);
        const auto parentOp = om.getSourceOp(inputTensor);
        if (parentOp->getOpType() != "DMATask" || parentOp->get<mv::DmaDirection>("direction") != mv::DmaDirectionEnum::DDR2NNCMX)
            continue;

        if (isException(parentOp, dmaOp))
            continue;

        const auto childOps = mv::findSinkLayers(dm, dmaOp->getOutputTensor(0));

        mv::linkNewOperationsReplacementRemoveFlows(mv::Data::OpListIterator(), inputTensor, om, dmaOp);
        if (mv::findSinkLayers(dm, inputTensor).size() == 1) {
            dmasToRemove.push_back(parentOp);
        } else {
            const auto parentInputTensor = parentOp->getInputTensor(0);

            for (auto childOp : childOps) {
                std::size_t inputSlot = 0;
                for (; inputSlot < childOp->inputSlots(); ++inputSlot)
                    if (childOp->getInputTensor(inputSlot)->getName() == inputTensor->getName())
                        break;
                auto inputFlow = childOp.leftmostInput();
                while(inputFlow != om.flowEnd())
                {
                    if (inputFlow->getTensor()->getName() == inputTensor->getName())
                        break;
                    ++inputFlow;
                }

                om.undefineFlow(inputFlow);
                childOp->setInputTensor(parentInputTensor, inputSlot, false);
                om.defineFlow(parentInputTensor, childOp, inputSlot);
            }
        }
    }

    for (auto& dmaOp : dmasToRemove) {
        mv::linkNewOperationsReplacementRemoveFlows(mv::Data::OpListIterator(), dmaOp->getInputTensor(0), om, dmaOp);
    }
}
