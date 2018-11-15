#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

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

mv::Data::OpListIterator linkNewOperationsReplacement(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j]);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
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

            opIt = linkNewOperationsReplacement(parentOpIt, conv2D, om, opIt);

        }

    }

}
