#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/common_functions.hpp"

static void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(FullyConnectedAsConv2D)
        .setFunc(fullyConnectedAsConv2DFcn)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "Replaces the fullyConnected op with conv2D using 1x1 kernels"
        );
    
    }

}

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == "FullyConnected")
        {
            
            pass.log(Logger::MessageType::Debug, "Found FullyConnected op " + opIt->getName());

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            auto sourceTensor = parentOpIt->getOutputTensor(0);
            std::cout << "parentOp name is" << parentOpIt->getName() << std::endl;
            std::cout << "source tesnor name is" << sourceTensor->getName() << std::endl;

            auto weightsData = opIt->getInputTensor(1)->getData();
            auto inputShape = sourceTensor->getShape();
            std::cout << "weightsData name is" << opIt->getInputTensor(1)->getName() << std::endl;
            
            Tensor weigthsTensor = *(opIt->getInputTensor(1));
            weigthsTensor.setOrder(Order(Order::getRowMajorID(weigthsTensor.getShape().ndims())));

            auto weights = om.constant(weigthsTensor.getData(), {inputShape[0], inputShape[1], inputShape[2], 
                opIt->getOutputTensor(0)->getShape()[1]}, sourceTensor->getDType(), 
                Order(Order::getRowMajorID(4)), opIt->getName() + "_weights");

            auto conv2D = om.conv(sourceTensor, weights, {1, 1}, {0, 0, 0, 0});
            pass.log(Logger::MessageType::Info, "Replaced FullyConnected op " + opIt->getName() + " with " + conv2D->getName());

            if (opIt->hasAttr("bias"))
            {
                auto biasTensorName = opIt->get<std::string>("bias");
                om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
                pass.log(Logger::MessageType::Info, "Moved Bias attribute of FullyConnected op " + opIt->getName() + " to " + conv2D->getName());
            }

            opIt = linkNewOperations(parentOpIt, conv2D, om, opIt);

        }

    }

}
