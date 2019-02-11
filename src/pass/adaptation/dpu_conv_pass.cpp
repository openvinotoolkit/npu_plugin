#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void DPUConvolutionFcn(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(DPUConvolution)
            .setFunc(DPUConvolutionFcn)
            .setGenre(PassGenre::Adaptation)
            .setDescription(
                "Replace all convolution operations with DPU tasks.\n"
                "Assume each convolution can be done with DPU on KMB.\n"
                "Assume each convolution should be done on DPU.");
    }
}

#define OUT_DMA 1 // 1=use CMX2DMA for output of dpuConv, 0=don't use DMA for output

void DPUConvolutionFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
{
    mv::OpModel om(model);
    mv::ControlModel cm(om);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Conv")
        {
            // ASSUMPTIONS:
            // - Convolution operation is not source nor sink of graph
            // - Convolution has exactly two inputs and one output
            // - Input tensor is produced by the only parent node
            // - Input tensor is attached as input channel #0
            // - Kernel (weights tensor) is input channel #1
            // - Output is attached to the only child node
            // - Input and output tensors reside in DDR

            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");
            auto name = opIt->getName();

            auto dmaInput = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX);
            auto dmaKernel = om.dMATask(kernel, mv::DmaDirectionEnum::DDR2CMX);

            auto dpuConv = om.dPUTaskConv({dmaInput, dmaKernel}, strides, padding, dilationFactor, "DPU_" + name);

            auto dmaInputFree = om.deAllocate(dmaInput, "FreeInput_DPU_" + name);
            auto dmaKernelFree = om.deAllocate(dmaKernel, "FreeKernel_DPU_" + name);

            auto dpuConvOp = om.getSourceOp(dpuConv);
            // This might be easy, but this does not work:
            /*********************************************
            auto dmaInputFreeOp = om.getSourceOp(dmaInputFree);
            auto dmaKernelFreeOp = om.getSourceOp(dmaKernelFree);
            ****************************************************/
            // So we need to go different way -- via naming of ops:
            auto dmaInputFreeOp = om.getOp("FreeInput_DPU_" + name);
            auto dmaKernelFreeOp = om.getOp("FreeKernel_DPU_" + name);

            cm.defineFlow(dpuConvOp, dmaInputFreeOp);
            cm.defineFlow(dpuConvOp, dmaKernelFreeOp);

        #if OUT_DMA
            auto newOutput = om.dMATask(dpuConv, mv::DmaDirectionEnum::CMX2DDR);
            auto newOutputFree = om.deAllocate(dpuConv, "FreeOutput_DPU_" + name);
            auto newOutputFreeOp = om.getOp("FreeOutput_DPU_" + name);
            auto newOutputOp = om.getSourceOp(newOutput);
            cm.defineFlow(newOutputOp, newOutputFreeOp);
        #else
            auto newOutput = dpuConv;
        #endif

            auto output = opIt.leftmostOutput();
            auto consumer = output.sink();
            auto slot = output->get<size_t>("sinkInput");
            std::cout << "output: "   << output->getName()     << std::endl; // DEBUG
            std::cout << "consumer: " << consumer->getName()   << std::endl; // DEBUG
            std::cout << "slot: "     << slot                  << std::endl; // DEBUG

            std::cout << "newOutput: " << newOutput->getName() << std::endl; // DEBUG
            consumer->setInputTensor(newOutput, slot, false);
            om.defineFlow(newOutput, consumer, slot);

            om.removeOp(opIt);
        }
    }

    std::cout << "Exiting DPU Convolution Pass"                << std::endl; // DEBUG
}
