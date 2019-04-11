#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void addDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void addFinalDMATaskFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void addDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

        MV_REGISTER_PASS(AddDeallocationTasks)
            .setFunc(addDeallocationTasksFcn)
            .setDescription(
               "Add deallocation tasks where needed in the Task graph");
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
    else if(opType.find("Constant") != std::string::npos)
        return false;
    else if(opType.find("Input") != std::string::npos)
        return false;
    else
        return true;
}

// Pass role: Add final DMA Task CMX2DDR (if needed)
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

// Pass role: Add DMA Task DDR2CMX where needed for each tensor input of Tasks.
// NOTE: This is the only DMA Pass that adds control flows (DMA dependencies)
void addDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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
    //dma_dependency = std::min(std::max(1, self.nn_cmx_memory/param.cluster_size), dma_dependency);
    //cluster size (memory of the tensor) = tensor dims multiplied * (data type /8)

    auto numCluster = 4;
    auto safetyFactor = 0.9;
    auto cmxSize = 3760128; //4MB in bytes.
    cmxSize /= numCluster;
    cmxSize *= safetyFactor;
    unsigned long _dma_dependency = 2;
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
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                auto inputOp = om.getSourceOp(inputTensor);
                if(!isTensorInCMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");

                    auto inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2CMX, "DMA"+inputOp->getName());
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

                    auto inputTensorDmaDimension = inputTensorDma->getShape().totalSize() * (inputTensorDma->getDType().getSizeInBits()/8);
                    dma_dependency = std::min(std::max((unsigned long)1, cmxSize/inputTensorDmaDimension), _dma_dependency + 1);
                    auto index = std::distance(sortedOps.begin(), std::find(sortedOps.begin(), sortedOps.end(), opIt));
                    if(index <= dma_dependency)
                        cm.defineFlow(om.getInput(), inputTensorDmaOp);
                    else
                        cm.defineFlow(sortedOps[index - dma_dependency], inputTensorDmaOp);
                }
            }
        }
    }
}

// Pass role: Add deallocation tasks for each Tensor
void addDeallocationTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    // Sorting ops in dataflow topological order. Will be needed later.
    auto sortedOps = om.topologicalSort();

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
        if(inputOp->getOpType() == "Constant" || inputOp->getOpType() == "ConstantInt" || inputOp->getOpType() == "ConstantDataElement")
            continue;

        auto opId = inputOp->get<unsigned>("opId");
        auto inputOpName = inputOp->getName();
        auto inputTensor = dataFlowIt->getTensor();

        std::string deallocationName("Deallocate"+inputOpName);

        // Each tensor must be deallocated once
        if(!om.checkOp(deallocationName))
        {
            // Flows names must be taken before the insertion of deallocation ops
            // Otherwise deallocation will appear as well
            auto flowsNames = inputTensor->get<std::set<std::string>>("flows");

            // Creating deallocation operation for the tensor and attaching it through a dataflow
            // to the operation that created it
            om.deallocate(inputTensor, deallocationName);
            auto deallocateInputOp = om.getOp(deallocationName);

            // Attaching also through a ControlFlow
            auto flowIt = cm.defineFlow(inputOp, deallocateInputOp);
            deallocateInputOp->set<unsigned>("opId", opId);

            // Control flows for the newly created operation must be attached now.
            // Checking all the ops that have this tensor as input

            std::vector<mv::Data::OpListIterator> sinkOperations;
            for(auto flowName : flowsNames)
            {
                auto df = dm.getDataFlow(flowName);
                sinkOperations.push_back(df.sink());
            }

            auto chosenOp = sortedOps.rbegin();
            for(; chosenOp != sortedOps.rend(); ++chosenOp)
                if(std::find(sinkOperations.begin(), sinkOperations.end(), *chosenOp) != sinkOperations.end())
                    break;

            if(!cm.checkControlFlow(*chosenOp, deallocateInputOp))
                cm.defineFlow(*chosenOp, deallocateInputOp);
            for(auto son = chosenOp->leftmostChild(); son != om.opEnd(); ++son)
                if(!cm.checkControlFlow(deallocateInputOp, son))
                    if(son->getOpType() != "Deallocate")
                        cm.defineFlow(deallocateInputOp, son);

        }
    }
}
