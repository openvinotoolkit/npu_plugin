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

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Conv")
        {
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            if (dilationFactor > 1)
            {

                /*Get the kernel attributes*/
                auto nonDialtedKernel = opIt->getInputTensor(1);
                auto nonDialtedKernelWidth = nonDialtedKernel->getShape()[0];
                auto nonDialtedKernelHeight = nonDialtedKernel->getShape()[1];
                auto nonDialtedKernelInputChannels = nonDialtedKernel->getShape()[2];
                auto nonDialtedKernelOutpuChannels = nonDialtedKernel->getShape()[3];
                auto nonDialtedKernelShape = nonDialtedKernel->getShape();
                auto nonDialtedKernelData = nonDialtedKernel->getIntData();


                /** Calculate dilated kernel shape
                  *
                  * dilatedWidth = kw + (kw - 1)(df - 1)
                  * dilatedHeight = kh + (kh - 1)(df - 1)
                  */
                mv::Shape dilatedKernelShape = mv::Shape({nonDialtedKernelWidth + (nonDialtedKernelWidth - 1) * (dilationFactor - 1),
                                                          nonDialtedKernelHeight + (nonDialtedKernelHeight - 1) * (dilationFactor - 1),
                                                          nonDialtedKernelInputChannels, nonDialtedKernelOutpuChannels});
                auto nonDialtedKernelOp = opIt.rightmostParent();
                unsigned currentOpId = nonDialtedKernelOp->get<unsigned>("opId");
                auto quantParams = nonDialtedKernelOp->get<mv::QuantizationParams>("quantParams");
                /*Populate dilated tensor with zeros*/
                std::vector<int64_t> defaultData(dilatedKernelShape.totalSize(), 0);

                std::array<unsigned short,4> padding = opIt->get< std::array<unsigned short,4> >("padding");
                if (padding[0])
                {
                    padding[0] = floor(nonDialtedKernelWidth/2);
                }
                if (padding[1])
                {
                    padding[1] = floor(nonDialtedKernelWidth/2);
                }
                if (padding[2])
                {
                    padding[2] = floor(nonDialtedKernelHeight/2);
                }
                if (padding[3])
                {
                    padding[3] = floor(nonDialtedKernelHeight/2);
                }
                opIt->set<std::array<unsigned short, 4>>("padding", {padding[0], padding[1], padding[2], padding[3]} );
                /*Create Dilated Kernel Tensor*/
                mv::Tensor dilatedKernel("dilatedKernel", dilatedKernelShape, nonDialtedKernel->getDType(), mv::Order(mv::Order::getRowMajorID(dilatedKernelShape.ndims())), defaultData);
                for (unsigned oc = 0; oc < dilatedKernelShape[3]; ++oc)
                    for (unsigned ic = 0; ic < dilatedKernelShape[2]; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < dilatedKernelShape[1]; ++kcolumn)
                            for (unsigned krow = 0; krow < dilatedKernelShape[0]; ++krow)
                                dilatedKernel.at({krow, kcolumn, ic, oc}) = quantParams.getZeroPoint(oc);

                for (unsigned oc = 0; oc < nonDialtedKernelOutpuChannels; ++oc)
                    for (unsigned ic = 0; ic < nonDialtedKernelInputChannels; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < nonDialtedKernelHeight; ++kcolumn)
                            for (unsigned krow = 0; krow < nonDialtedKernelWidth; ++krow)
                                /*Copy non-dilated weights into the dilated kernel*/
                                if (krow != 0 || kcolumn != 0)
                                    dilatedKernel.at({krow + (dilationFactor - 1) * krow, kcolumn + (dilationFactor - 1) * kcolumn, ic, oc}) = nonDialtedKernel->at({krow, kcolumn, ic, oc});
                                else
                                    dilatedKernel.at({krow, kcolumn, ic, oc}) = nonDialtedKernel->at({krow, kcolumn, ic, oc});

                auto dilatedConstant = om.constantDataElement(
                    dilatedKernel.getData(),
                    dilatedKernelShape,
                    dilatedKernel.getDType(),
                    dilatedKernel.getOrder(),
                    {{},{},{},{}},
                    nonDialtedKernelOp->getName() + "_Dilated");

                auto DialtedKernelData = dilatedConstant->getIntData();

                om.removeOp(nonDialtedKernelOp);
                    
                om.defineFlow(dilatedConstant, opIt, 1);
                opIt->set<std::array<unsigned short, 2>>("kSize", {dilatedKernelShape[0], dilatedKernelShape[1]} );
                opIt->setInputTensor(dilatedConstant, 1);
                auto DialtedKernelOp = opIt.rightmostParent();
                DialtedKernelOp->set<unsigned>("opId", currentOpId);
                DialtedKernelOp->set<mv::QuantizationParams>("quantParams", quantParams);
                DialtedKernelOp->getOutputTensor()[0]->set<mv::QuantizationParams>("quantParams", quantParams);
            }

        }
        
    }
    
}
