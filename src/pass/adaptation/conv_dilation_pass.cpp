#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"

static void convDilationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvolutionDilation)
            .setFunc(convDilationFcn)
            .setDescription(
                "This pass dilates a kernel");
    }
}

void convDilationFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv")
        {
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            if (dilationFactor > 1)
            {

                /*Get the kernel attributes*/
                auto nonDilatedKernel = opIt->getInputTensor(1);
                auto nonDilatedKernelWidth = nonDilatedKernel->getShape()[KERNEL_WIDTH];
                auto nonDilatedKernelHeight = nonDilatedKernel->getShape()[KERNEL_HEIGHT];
                auto nonDilatedKernelInputChannels = nonDilatedKernel->getShape()[KERNEL_INPUT_CHANNELS];
                auto nonDilatedKernelOutpuChannels = nonDilatedKernel->getShape()[KERNEL_OUTPUT_CHANNELS];
                auto nonDilatedKernelShape = nonDilatedKernel->getShape();


                /** Calculate dilated kernel shape
                  *
                  * dilatedWidth = kw + (kw - 1)(df - 1)
                  * dilatedHeight = kh + (kh - 1)(df - 1)
                  */
                mv::Shape dilatedKernelShape = mv::Shape({nonDilatedKernelWidth + (nonDilatedKernelWidth - 1) * (dilationFactor - 1),
                                                          nonDilatedKernelHeight + (nonDilatedKernelHeight - 1) * (dilationFactor - 1),
                                                          nonDilatedKernelInputChannels, nonDilatedKernelOutpuChannels});
                auto nonDilatedKernelOp = opIt.rightmostParent();
                unsigned currentOpId = nonDilatedKernelOp->get<unsigned>("opId");
                auto quantParams = nonDilatedKernelOp->get<mv::QuantizationParams>("quantParams");
                /*Populate dilated tensor with zeros*/

                /*Create Dilated Kernel Tensor*/

                //build the dilated kernel with zero points corresponding to each channel - KMB does not support different zp per channel
                std::vector<int64_t> defaultData(dilatedKernelShape.totalSize(), quantParams.getZeroPoint(0));
                mv::Tensor dilatedKernel("dilatedKernel", dilatedKernelShape, nonDilatedKernel->getDType(), mv::Order(mv::Order::getRowMajorID(dilatedKernelShape.ndims())), defaultData);

                for (unsigned oc = 0; oc < nonDilatedKernelOutpuChannels; ++oc)
                    for (unsigned ic = 0; ic < nonDilatedKernelInputChannels; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < nonDilatedKernelHeight; ++kcolumn)
                            for (unsigned krow = 0; krow < nonDilatedKernelWidth; ++krow)
                                /*Copy non-dilated weights into the dilated kernel*/
                                if (krow != 0 || kcolumn != 0)
                                    dilatedKernel.at({krow + (dilationFactor - 1) * krow, kcolumn + (dilationFactor - 1) * kcolumn, ic, oc}) = nonDilatedKernel->at({krow, kcolumn, ic, oc});
                                else
                                    dilatedKernel.at({krow, kcolumn, ic, oc}) = nonDilatedKernel->at({krow, kcolumn, ic, oc});

                auto dilatedKernelOp = om.constantDataElement(
                    dilatedKernel.getData(),
                    dilatedKernelShape,
                    dilatedKernel.getDType(),
                    dilatedKernel.getOrder(),
                    quantParams,
                    nonDilatedKernelOp->getName() + "_Dilated");

                om.removeOp(nonDilatedKernelOp);
                om.defineFlow(dilatedKernelOp, opIt, 1);
                opIt->set<std::array<unsigned short, 2>>("kSize", {dilatedKernelShape[KERNEL_WIDTH], dilatedKernelShape[KERNEL_HEIGHT]} );
                opIt->setInputTensor(dilatedKernelOp, 1, false);
                opIt->set<unsigned>("dilationFactor", 1);
                auto DilatedKernelOpFetched = opIt.rightmostParent();
                DilatedKernelOpFetched->set<unsigned>("opId", currentOpId);
            }

        }

    }

}
