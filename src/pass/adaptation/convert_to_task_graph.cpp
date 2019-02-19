#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void ConvertToTaskGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);


namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvertToTaskGraph)
            .setFunc(ConvertToTaskGraphFcn)
            .setDescription(
                "Replace all convolution operations with DPU tasks.\n"
                "Assume each convolution can be done with DPU on KMB.\n"
                "Assume each convolution should be done on DPU.");
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

mv::Data::TensorIterator addWeightsTable(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& kernelWeightsTableName, unsigned outputChannels)
{
    std::vector<double> weightTableData(4 * outputChannels, 0);
    auto weightTable = om.constant(weightTableData, {outputChannels, 1, 1, 4}, mv::DType("UInt32"), mv::Order("WHCN"), kernelWeightsTableName);
    om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    weightTable = om.dMATask(weightTable, mv::DmaDirectionEnum::DDR2CMX);
    dpuTaskOp->addInputTensor(weightTable);

    return weightTable;
}

//TODO: Copy OpId, but Ian is needed in this case.
void ConvertToTaskGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // TODO/WIP: Add weights table as well

    // Pass main assumption is that we are working on the original graph
    // So until we start modifing the graph there are no DMA tasks involved

    // While loop is preferred in a loop like this were we are performing eliminations
    // as it gives more flexibility on when to increment the iterator
    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        if (opIt->getOpType() == "Conv")
        {
            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);
            auto inputOpName = om.getSourceOp(input)->getName();
            auto kernelOpName = om.getSourceOp(kernel)->getName();
            auto opId = opIt->get<unsigned>("opId");
            auto inputOpId = om.getSourceOp(input)->get<unsigned>("opId");
            auto kernelOpId = om.getSourceOp(kernel)->get<unsigned>("opId");

            std::string inputDeallocationName("Deallocate"+inputOpName);
            std::string kernelDeallocationName("Deallocate"+kernelOpName);
            std::string kernelWeightsTableName(kernelOpName + "WeightsTable");
            std::string kernelWeightsTableDeallocationName("Deallocate"+kernelWeightsTableName);

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");
            auto name = opIt->getName();

            // NOTE: This check is always needed, regardless of the above assumption
            if(!isTensorInCMX(input, om))
            {
                input = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);
                om.getSourceOp(input)->set<unsigned>("opId", inputOpId);
            }


            // NOTE: This check is not actually needed given the above assumption, as it will always be true.
            //if(!isTensorInCMX(kernel, om))
                //kernel = om.dMATask(kernel, mv::DmaDirectionEnum::DDR2CMX);

            kernel = om.dMATask(kernel, mv::DmaDirectionEnum::DDR2CMX);
            om.getSourceOp(kernel)->set<unsigned>("opId", kernelOpId);

            auto dpuConv = om.dPUTaskConv({input, kernel}, strides, padding, dilationFactor, "DPU_" + name);
            auto dpuConvOp = om.getSourceOp(dpuConv);
            dpuConvOp->set<unsigned>("opId", opId);

            // Let's take the data we need for Weights Table (WT)
            auto weightTable = addWeightsTable(om, dpuConvOp, kernelWeightsTableName, kernel->getShape()[3]);
            om.getSourceOp(weightTable)->set<unsigned>("opId", opId);

            // DPUConvOp input slots status: 0 - Input, 1 - Weights, 2 - Weights Table
            om.defineFlow(weightTable, dpuConvOp, 2);

            om.deallocateTask(input, inputDeallocationName);
            om.deallocateTask(kernel, kernelDeallocationName);
            om.deallocateTask(weightTable, kernelWeightsTableDeallocationName);

            auto dmaInputFreeOp = om.getOp(inputDeallocationName);
            dmaInputFreeOp->set<unsigned>("opId", inputOpId);
            auto dmaKernelFreeOp = om.getOp(kernelDeallocationName);
            dmaKernelFreeOp->set<unsigned>("opId", kernelOpId);
            auto dmaKernelWeightsTableFreeOp = om.getOp(kernelWeightsTableDeallocationName);
            dmaKernelWeightsTableFreeOp->set<unsigned>("opId", opId);

            cm.defineFlow(dpuConvOp, dmaInputFreeOp);
            cm.defineFlow(dpuConvOp, dmaKernelFreeOp);
            cm.defineFlow(dpuConvOp, dmaKernelWeightsTableFreeOp);

            for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
            {
                auto consumer = output.sink();
                auto slot = output->get<size_t>("sinkInput");
                consumer->setInputTensor(dpuConv, slot, false);
                om.defineFlow(dpuConv, consumer, slot);
            }

            auto backup = opIt;
            ++opIt;
            om.removeOp(backup);
        }
        if (opIt->getOpType() == "MaxPool")
        {
            auto input = opIt->getInputTensor(0);
            auto inputOpName = om.getSourceOp(input)->getName();

            auto opId = opIt->get<unsigned>("opId");
            auto inputOpId = om.getSourceOp(input)->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto name = opIt->getName();

            if(!isTensorInCMX(input, om))
            {
                input = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);
                om.getSourceOp(input)->set<unsigned>("opId", inputOpId);
            }

            auto dpuPool = om.dPUTaskMaxPool({input}, kernelSize, strides, padding, "DPU_" + name);
            auto dpuPoolOp = om.getSourceOp(dpuPool);
            dpuPoolOp->set<unsigned>("opId", opId);

            // Let's take the data we need for Weights Table (WT)
            std::string kernelWeightsTableName(opIt->getName() + "WeightsTable");
            auto weightTable = addWeightsTable(om, dpuPoolOp, kernelWeightsTableName, input->getShape()[2]);
            om.getSourceOp(weightTable)->set<unsigned>("opId", opId);

            // DPUPoolOp input slots status: 0 - Input, 1 Weights Table
            om.defineFlow(weightTable, dpuPoolOp, 1);

            std::string inputDeallocationName("Deallocate"+inputOpName);
            std::string kernelWeightsTableDeallocationName("Deallocate"+kernelWeightsTableName);

            om.deallocateTask(input, inputDeallocationName);
            om.deallocateTask(weightTable, kernelWeightsTableDeallocationName);

            auto dmaInputFreeOp = om.getOp(inputDeallocationName);
            dmaInputFreeOp->set<unsigned>("opId", inputOpId);

            auto dmaKernelWeightsTableFreeOp = om.getOp(kernelWeightsTableDeallocationName);
            dmaKernelWeightsTableFreeOp->set<unsigned>("opId", opId);

            cm.defineFlow(dpuPoolOp, dmaInputFreeOp);
            cm.defineFlow(dpuPoolOp, dmaKernelWeightsTableFreeOp);

            for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
            {
                auto consumer = output.sink();
                auto slot = output->get<size_t>("sinkInput");
                consumer->setInputTensor(dpuPool, slot, false);
                om.defineFlow(dpuPool, consumer, slot);
            }

            auto backup = opIt;
            ++opIt;
            om.removeOp(backup);
        }

        if (opIt->getOpType() == "AveragePool")
        {
            auto input = opIt->getInputTensor(0);
            auto inputOpName = om.getSourceOp(input)->getName();

            auto opId = opIt->get<unsigned>("opId");
            auto inputOpId = om.getSourceOp(input)->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto name = opIt->getName();

            if(!isTensorInCMX(input, om))
            {
                input = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);
                om.getSourceOp(input)->set<unsigned>("opId", inputOpId);
            }

            auto dpuPool = om.dPUTaskAveragePool({input}, kernelSize, strides, padding, "DPU_" + name);
            auto dpuPoolOp = om.getSourceOp(dpuPool);
            dpuPoolOp->set<unsigned>("opId", opId);

            // Let's take the data we need for Weights Table (WT)
            std::string kernelWeightsTableName(opIt->getName() + "WeightsTable");
            auto weightTable = addWeightsTable(om, dpuPoolOp, kernelWeightsTableName, input->getShape()[2]);
            om.getSourceOp(weightTable)->set<unsigned>("opId", opId);

            // DPUPoolOp input slots status: 0 - Input, 1 Weights Table
            om.defineFlow(weightTable, dpuPoolOp, 1);

            std::string inputDeallocationName("Deallocate"+inputOpName);
            std::string kernelWeightsTableDeallocationName("Deallocate"+kernelWeightsTableName);

            om.deallocateTask(input, inputDeallocationName);
            om.deallocateTask(weightTable, kernelWeightsTableDeallocationName);

            auto dmaInputFreeOp = om.getOp(inputDeallocationName);
            dmaInputFreeOp->set<unsigned>("opId", inputOpId);

            auto dmaKernelWeightsTableFreeOp = om.getOp(kernelWeightsTableDeallocationName);
            dmaKernelWeightsTableFreeOp->set<unsigned>("opId", opId);

            cm.defineFlow(dpuPoolOp, dmaInputFreeOp);
            cm.defineFlow(dpuPoolOp, dmaKernelWeightsTableFreeOp);

            for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
            {
                auto consumer = output.sink();
                auto slot = output->get<size_t>("sinkInput");
                consumer->setInputTensor(dpuPool, slot, false);
                om.defineFlow(dpuPool, consumer, slot);
            }

            auto backup = opIt;
            ++opIt;
            om.removeOp(backup);
        }

        if (opIt->getOpType() == "Output")
        {
            auto input = opIt->getInputTensor(0);

            auto opId = opIt->get<unsigned>("opId");
            std::string oldOutputName(opIt->getName());

            if(isTensorInCMX(input, om))
            {
                auto newInput = om.dMATask(input, mv::DmaDirectionEnum::CMX2DDR);
                om.getSourceOp(newInput)->set<unsigned>("opId", opId);
                auto backup = opIt;
                ++opIt;
                om.removeOp(backup);
                om.output(newInput, oldOutputName);
                om.getOp(oldOutputName)->set<unsigned>("opId", opId);
            }
            else
                ++opIt;
        }
        else
            ++opIt;
    }
}
