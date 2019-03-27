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
    auto inputOp = om.getSourceOp(input);

    auto opId = opIt->get<unsigned>("opId");
    std::string oldOutputName(opIt->getName());

    if(isTensorInCMX(input, om))
    {
        auto newInput = om.dMATask(input, mv::DmaDirectionEnum::CMX2DDR, "DMA"+inputOp->getName());
        auto newInputOp = om.getSourceOp(newInput);
        newInputOp->set<unsigned>("opId", opId);
        auto backup = opIt;
        om.removeOp(backup);
        om.output(newInput, oldOutputName);
        auto newOutputOp = om.getOp(oldOutputName);
        newOutputOp->set<unsigned>("opId", opId);
    }
}

void addDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

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
    //dma_dependency = std::min(std::max(1, self.nn_cmx_memory/param.cluster_size), dma_dependency);
    //cluster size (memory of the tensor) = tensor dims multiplied * (data type /8)

    auto numCluster = 4;
    auto safetyFactor = 0.9;
    auto cmxSize = 4 * 1024 * 1024; //4MB in bytes.
    cmxSize /= numCluster;
    cmxSize *= safetyFactor;
    unsigned long _dma_dependency = 5;
    int dma_dependency;

    // Pass main assumption is that we are working on the original graph, just with the Ops converted to DPUTasks
    // We don't need to perform eliminations in this pass, we can use a for loop to iterate among operations
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "DPUTask")
        {
            std::cout << "DPUTask name is " << opIt->getName() << std::endl;
            auto opId = opIt->get<unsigned>("opId");
            auto flow = opIt.leftmostInput();
            for(unsigned i = 0; i < opIt->inputSlots(); ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                std::cout << "Input tensor name is " << inputTensor->getName() << std::endl;
                auto inputOp = om.getSourceOp(inputTensor);
                auto inputOpName = inputOp->getName();
                std::cout << "Input OP name is " << inputOpName << std::endl;
                std::string inputDeallocationName("Deallocate"+inputOpName);
                if(!isTensorInCMX(inputTensor, om))
                {
                    auto inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2CMX, "DMA"+inputOp->getName());
                    auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                    inputTensorDmaOp->set<unsigned>("opId", opId);

                    auto backupFlow = flow;
                    ++flow;
                    om.undefineFlow(backupFlow);
                    opIt->setInputTensor(inputTensorDma, i, false);
                    om.defineFlow(inputTensorDmaOp, 0, opIt, i);


                    auto inputTensorDmaDimension = inputTensorDma->getShape().totalSize() * (inputTensorDma->getDType().getSizeInBits()/8);
                    dma_dependency = std::min(std::max((unsigned long)1, cmxSize/inputTensorDmaDimension), _dma_dependency);
                    auto index = std::distance(sortedOps.begin(), std::find(sortedOps.begin(), sortedOps.end(), opIt));
                    std::cout << "Index for layer " << opIt->getName() << "is " << index << std::endl;
                    if(index <= dma_dependency) {
                        std::cout << "connecting " << inputOpName << "to " << "input" << std::endl;
                        cm.defineFlow(om.getInput(), inputTensorDmaOp);
                    }
                    else{
                        std::cout << "not connecting " << inputOpName << "to " << "input" << std::endl;
                        cm.defineFlow(sortedOps[index - dma_dependency], inputTensorDmaOp);
                    inputTensor = inputTensorDma;
                    }
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
