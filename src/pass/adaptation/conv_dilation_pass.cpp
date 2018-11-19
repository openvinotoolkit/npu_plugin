#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"

static void convDilation(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &);

namespace mv
{

namespace pass
{
MV_REGISTER_PASS(ConvolutionDilation)
    .setFunc(convDilation)
    .setGenre(PassGenre::Adaptation)
    .setDescription(
        "This pass dilates a kernel");
}
} // namespace mv

void convDilation(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
{

    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "ConvDilated")
        {
            /*Get the kernel attributes*/
            auto nonDialtedKernel = opIt->getInputTensor(1);
            auto nonDialtedKernelWidth = nonDialtedKernel->get<mv::Shape>("shape")[0];
            auto nonDialtedKernelKernelHeight = nonDialtedKernel->get<mv::Shape>("shape")[1];
            auto nonDialtedKernelKernelInputChannels = nonDialtedKernel->get<mv::Shape>("shape")[2];
            auto nonDialtedKernelKernelOutpuChannels = nonDialtedKernel->get<mv::Shape>("shape")[3];
            auto dilationFactor = opIt->get<unsigned>("dilation");

            std::cout << "Dilation factor " << dilationFactor << std::endl;
            std::cout << "Non dilated shape: " << nonDialtedKernel->getShape().toString() << std::endl;

            /** Calculate dilated kernel shape
              *
              * dilatedWidth = kw + (kw - 1)(df - 1)
              * dilatedHeight = kh + (kh - 1)(df - 1)
              *
              */

            mv::Shape dilatedKernelShape = mv::Shape({nonDialtedKernelWidth + (nonDialtedKernelWidth - 1) * (dilationFactor - 1),
                                                      nonDialtedKernelWidth + (nonDialtedKernelWidth - 1) * (dilationFactor - 1),
                                                      nonDialtedKernel->get<mv::Shape>("shape")[2], nonDialtedKernel->get<mv::Shape>("shape")[3]});

            std::cout << "New shape: " << dilatedKernelShape.toString() << std::endl;

            std::vector<double> defaultData(dilatedKernelShape.totalSize(), 0);

            mv::Tensor dilatedKernel("dilatedKernel", dilatedKernelShape, nonDialtedKernel->getDType(), mv::Order(mv::Order::getRowMajorID(dilatedKernelShape.ndims())), defaultData);

       

            for (unsigned oc = 0; oc < nonDialtedKernelKernelOutpuChannels; ++oc)
            {
                for (unsigned ic = 0; ic < nonDialtedKernelKernelInputChannels; ++ic)
                {
                    for (unsigned ky = 0; ky < nonDialtedKernelKernelHeight; ++ky)
                    {

                        for (unsigned kx = 0; kx < nonDialtedKernelWidth; ++kx)
                        {

                            std::cout << "Non dilated kernel index is         " << kx << ", " << ky << ", " << ic << ", " << oc << std::endl;

                            if (kx != 0 || ky != 0)
                            {
                                std::cout << "Dilated kernel index is (x !=0)     " << kx + (dilationFactor -1)*kx << ", " << ky + (dilationFactor -1)*ky << ", " << ic << ", " << oc << std::endl;
                                dilatedKernel.at({kx + (dilationFactor -1)*kx, ky + (dilationFactor -1)*ky, ic, oc}) = nonDialtedKernel->at({kx, ky, ic, oc});
                            }
                            
                            else
                            {
                                std::cout << "Dilated kernel index is (x=0)       " << kx << ", " << ky << ", " << ic << ", " << oc << std::endl;
                                dilatedKernel.at({kx, ky, ic, oc}) = nonDialtedKernel->at({kx, ky, ic, oc});
                            }

                            
                        }
                    }
                }
            }

            //new_data = backup_tensor.getData();
        }
    }

    std::cout << "exiting dilation pass " << std::endl;
    exit(1);
}
