#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void ConvertToTaskGraphFcn(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvertToTaskGraph)
            .setFunc(ConvertToTaskGraphFcn)
            .setGenre(PassGenre::Adaptation)
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

//TODO: Copy OpId, but Ian is needed in this case.
void ConvertToTaskGraphFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
{
    mv::OpModel om(model);
    mv::ControlModel cm(om);

    // While loop is preferred in a loop like this were we are performing eliminations
    // as it gives more flexibility on when to increment the iterator
    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        if (opIt->getOpType() == "Conv")
        {
            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");
            auto name = opIt->getName();

            if(!isTensorInCMX(input, om))
                input = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);
            if(!isTensorInCMX(kernel, om))
                kernel = om.dMATask(kernel, mv::DmaDirectionEnum::DDR2CMX);

            auto dpuConv = om.dPUTaskConv({input, kernel}, strides, padding, dilationFactor, "DPU_" + name);

            auto inputOpName = om.getSourceOp(input)->getName();
            auto kernelOpName = om.getSourceOp(kernel)->getName();
            std::string inputDeallocationName("Deallocate"+inputOpName);
            std::string kernelDeallocationName("Deallocate"+kernelOpName);

            om.deAllocate(input, inputDeallocationName);
            om.deAllocate(kernel, kernelDeallocationName);

            auto dpuConvOp = om.getSourceOp(dpuConv);
            auto dmaInputFreeOp = om.getOp(inputDeallocationName);
            auto dmaKernelFreeOp = om.getOp(kernelDeallocationName);

            cm.defineFlow(dpuConvOp, dmaInputFreeOp);
            cm.defineFlow(dpuConvOp, dmaKernelFreeOp);

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

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto name = opIt->getName();

            if(!isTensorInCMX(input, om))
                input = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);

            auto dpuPool = om.dPUTaskMaxPool({input}, kernelSize, strides, padding, "DPU_" + name);

            auto inputOpName = om.getSourceOp(input)->getName();
            std::string inputDeallocationName("Deallocate"+inputOpName);
            om.deAllocate(input, inputDeallocationName);

            auto dpuPoolOp = om.getSourceOp(dpuPool);
            auto dmaInputFreeOp = om.getOp(inputDeallocationName);

            cm.defineFlow(dpuPoolOp, dmaInputFreeOp);

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

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto name = opIt->getName();

            if(!isTensorInCMX(input, om))
                input = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);

            auto dpuPool = om.dPUTaskAveragePool({input}, kernelSize, strides, padding, "DPU_" + name);

            auto inputOpName = om.getSourceOp(input)->getName();
            std::string inputDeallocationName("Deallocate"+inputOpName);
            om.deAllocate(input, inputDeallocationName);

            auto dpuPoolOp = om.getSourceOp(dpuPool);
            auto dmaInputFreeOp = om.getOp(inputDeallocationName);

            cm.defineFlow(dpuPoolOp, dmaInputFreeOp);

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
            if(isTensorInCMX(input, om))
            {
                auto newInput = om.dMATask(input, mv::DmaDirectionEnum::CMX2DDR);
                auto backup = opIt;
                ++opIt;
                om.removeOp(backup);
                om.output(newInput);
            }
            else
                ++opIt;
        }
        else
            ++opIt;
    }
}
