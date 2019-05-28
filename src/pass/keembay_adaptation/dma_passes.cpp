#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void addWeightsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&);
static void addInitialAndFinalDMATaskFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddWeightsDMATasks)
            .setFunc(addWeightsDMATasksFcn)
            .setDescription(
               "Add Weights DMA Tasks where needed in the Task graph");

        MV_REGISTER_PASS(AddInitialAndFinalDMATask)
            .setFunc(addInitialAndFinalDMATaskFcn)
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
    else if(opType == "WeightsTable")
        return false;
    else if(opType == "SparsityMap")
        return false;
    else if(opType == "Input")
        return false;
    else
        return true;
}

// Pass role: Add initial and final DMA Task CMX2DDR (if needed)
void addInitialAndFinalDMATaskFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // INPUT
    auto inputOp = om.getInput();
    auto inputTensor = inputOp->getOutputTensor(0);

    auto opId = inputOp->get<unsigned>("opId");
    mv::QuantizationParams quantParams = {{},{},{},{}};
    if(inputTensor->hasAttr("quantParams"))
        quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
    if(!isTensorInCMX(inputTensor, om))
    {
        auto flows = inputTensor->get<std::set<std::string>>("flows");

        auto inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2CMX,quantParams, inputOp->getName()+"_DDR2CMX");
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

    // OUTPUT
    auto opIt = om.getOutput();
    auto input = opIt->getInputTensor(0);
    inputOp = om.getSourceOp(input);

    opId = opIt->get<unsigned>("opId");
    std::string oldOutputName(opIt->getName());
    quantParams = {{},{},{},{}};
    if(input->hasAttr("quantParams"))
        quantParams = input->get<mv::QuantizationParams>("quantParams");
    if(isTensorInCMX(input, om))
    {
        auto newInput = om.dMATask(input, mv::DmaDirectionEnum::CMX2DDR, quantParams, inputOp->getName()+"_CMX2DDR");
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
void addWeightsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto removeOps = [] (std::vector<mv::Data::OpListIterator>& list, const std::string& opType)
    {
        list.erase(std::remove_if(list.begin(), list.end(), [opType](mv::Data::OpListIterator it) { return it->getOpType() == opType;}), list.end());
    };

    auto sortedOps = om.topologicalSort();
    removeOps(sortedOps, "Constant");
    removeOps(sortedOps, "ConstantInt");
    removeOps(sortedOps, "ConstantDataElement");

    // How self.nn_cmx_memory is computed
    //    cmxSize = 4194304;
    //    4194304 / 4 (cluster) = 1048576;
    //    0.9 (safety_factor) * 1048576 = 943718.4
    //
    //dma_dependency = std::min(std::max(1, self.nn_cmx_memory/param.cluster_size), dma_dependency);
    //    This is the weights prefetch number. It specifies how early to start the inbound DMA for weights.
    //    The units are number of ops preceeding current conv in the topographically sorted ops list.
    //    If the weights tensor is very large (eg > 1/2 of CMX) then the specified prefetch parameter (eg 2)
    //    would be reduced. This assumes that the only fit partial serialization would find for such a
    //    big weights tensor would be to start the DMA right before it is needed. For smaller tensors, the
    //    user-specified prefetch number will be used. The prefetch edge added is for partial serialization.
    //

    auto globalConfigParams = model.getGlobalConfigParams();
    auto cmxSize = globalConfigParams->get<unsigned>("cmx");

    int _dma_dependency = passDesc.get<int>("weights_prefetch");
    int dma_dependency;

    // Pass main assumption is that we are working on the original graph, just with the Ops converted to DPUTasks
    // We don't need to perform eliminations in this pass, we can use a for loop to iterate among operations
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "DPUTask")
        {
            auto opId = opIt->get<unsigned>("opId");
            unsigned n = opIt->inputSlots();
            unsigned inputOutputTensors = 0;
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                if(!inputTensor->isPopulated())
                {
                    ++inputOutputTensors;
                    continue;
                }
                mv::QuantizationParams quantParams = {{},{},{},{}};
                if(inputTensor->hasAttr("quantParams"))
                    quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
                auto inputOp = om.getSourceOp(inputTensor);
                if(!isTensorInCMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");

                    auto inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2CMX,quantParams, inputOp->getName()+"_DDR2CMX");
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

                    //NOTE: This will change with multicluster
                    long unsigned inputTensorDmaDimension = inputTensorDma->sizeBytes();
                    for(unsigned j = 0; j < inputOutputTensors; ++j)
                        inputTensorDmaDimension += opIt->getInputTensor(j)->sizeBytes();

                    int partsPerCMX = std::max((unsigned long)1, cmxSize/inputTensorDmaDimension);
                    if (partsPerCMX < (_dma_dependency + 1))
                    {
                        dma_dependency = partsPerCMX;
                        pass.log(mv::Logger::MessageType::Warning, "Overriding weights prefetch parameter due to large tensor DMA"+inputOp->getName()+" : using " + std::to_string(partsPerCMX) + " vs " + std::to_string(_dma_dependency+1));
                    }
                    else
                        dma_dependency =  _dma_dependency + 1 ;


                    auto index = std::distance(sortedOps.begin(), std::find(sortedOps.begin(), sortedOps.end(), opIt));
                    if(index <= dma_dependency)
                        cm.defineFlow(om.getInput(), inputTensorDmaOp);
                    else
                    {
                        cm.defineFlow(sortedOps[index - dma_dependency], inputTensorDmaOp);
                        inputTensor = inputTensorDma;
                    }
                }
            }
        }
    }
}

