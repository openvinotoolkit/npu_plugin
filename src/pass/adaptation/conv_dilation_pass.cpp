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
std::cout<< __FUNCTION__ << ":" << __LINE__ <<std::endl;

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


                /** Calculate dilated kernel shape
                  *
                  * dilatedWidth = kw + (kw - 1)(df - 1)
                  * dilatedHeight = kh + (kh - 1)(df - 1)
                  *
                  */

                mv::Shape dilatedKernelShape = mv::Shape({nonDialtedKernelWidth + (nonDialtedKernelWidth - 1) * (dilationFactor - 1),
                                                          nonDialtedKernelHeight + (nonDialtedKernelHeight - 1) * (dilationFactor - 1),
                                                          nonDialtedKernelInputChannels, nonDialtedKernelOutpuChannels});
                /*Populate dilated tensor with zeros*/
                std::vector<int64_t> defaultData(dilatedKernelShape.totalSize(), 0);

                /*Create Tensor*/
                mv::Tensor dilatedKernel("dilatedKernel", dilatedKernelShape, nonDialtedKernel->getDType(), mv::Order(mv::Order::getRowMajorID(dilatedKernelShape.ndims())), defaultData);

                for (unsigned oc = 0; oc < nonDialtedKernelOutpuChannels; ++oc)
                    for (unsigned ic = 0; ic < nonDialtedKernelInputChannels; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < nonDialtedKernelHeight; ++kcolumn)
                            for (unsigned krow = 0; krow < nonDialtedKernelWidth; ++krow)
                                /*Copy non-dilated weights into the dilated kernel*/
                                if (krow != 0 || kcolumn != 0)
                                    dilatedKernel.at({krow + (dilationFactor - 1) * krow, kcolumn + (dilationFactor - 1) * kcolumn, ic, oc}) = nonDialtedKernel->at({krow, kcolumn, ic, oc});
                                else
                                    dilatedKernel.at({krow, kcolumn, ic, oc}) = nonDialtedKernel->at({krow, kcolumn, ic, oc});

                auto nonDialtedKernelOp = opIt.rightmostParent();
                auto quantParams = nonDialtedKernelOp->get<mv::QuantizationParams>("quantParams");
                unsigned currentOpId = nonDialtedKernelOp->get<unsigned>("opId");

                auto dilatedConstant = om.constantDataElement(
                    dilatedKernel.getData(),
                    dilatedKernelShape,
                    dilatedKernel.getDType(),
                    dilatedKernel.getOrder(),
                    {{},{},{},{}},
                    nonDialtedKernelOp->getName() + "_Dilated");
                om.removeOp(nonDialtedKernelOp);




                std::cout << "############# nonDialtedKernel"<< std::endl;
                std::cout << "############# nonDialtedKernel size = "<< nonDialtedKernel->size()<< std::endl;

                for (unsigned oc = 0; oc < nonDialtedKernelOutpuChannels; ++oc)
                    for (unsigned ic = 0; ic < nonDialtedKernelInputChannels; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < nonDialtedKernelHeight; ++kcolumn)
                            for (unsigned krow = 0; krow < nonDialtedKernelWidth; ++krow)
                                    std::cout <<"element[ "<<std::dec<<krow <<"," << kcolumn <<"," << ic <<"," << oc <<"]=" << nonDialtedKernel->at({krow, kcolumn, ic, oc}).i64_()<< std::endl;

                // //std::cout << nonDialtedKernel->toString() << std::endl;
                // for (int i=0; i< nonDialtedKernel->size(); i++)
                //     //std::cout <<"counter i=" << i << " el=" << nonDialtedKernel->at(i).i64_() << std::endl;
                //     std::cout <<"counter i=" << std::dec << i << " el=" << nonDialtedKernel->getDataPacked()[i] << std::endl;

                std::cout << "############# dilatedKernel"<< std::endl;
                //std::cout << dilatedKernel.toString() << std::endl;
                std::cout << "############# nonDialtedKernel size = "<< dilatedKernel.size()<< std::endl;

for (unsigned oc = 0; oc < dilatedKernelShape[0]; ++oc)
                    for (unsigned ic = 0; ic < dilatedKernelShape[1]; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < dilatedKernelShape[2]; ++kcolumn)
                            for (unsigned krow = 0; krow < dilatedKernelShape[3]; ++krow)
                                    std::cout <<"element[ "<<std::dec<<krow <<"," << kcolumn <<"," << ic <<"," << oc <<"]=" << dilatedKernel.at({krow, kcolumn, ic, oc}).i64_()<< std::endl;


                // for (int i=0; i< dilatedKernel.size(); i++)
                //     //std::cout <<"counter i=" << i << " el="  << dilatedKernel.at(i).i64_() << std::endl;
                //     std::cout <<"counter i=" << std::dec  << i << " el="  << dilatedKernel.getDataPacked()[i] << std::endl;

                // //for (auto el=dilatedKernel.getData().begin(); el!= dilatedKernel.getData().end(); el++)
                // //    std::cout <<"counter i=" << el->i64_() << std::endl;
                    
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
