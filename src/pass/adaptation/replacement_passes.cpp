#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
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

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::FullyConnected)
        {
            
            pass.log(Logger::MessageType::Debug, "Found FullyConnected op " + opIt->getName());

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            auto sourceTensor = parentOpIt->getOutputTensor(0);

            auto weightsData = opIt->getInputTensor(1)->getData();
            auto inputShape = sourceTensor->getShape();

            Tensor weigthsTensor = *(opIt->getInputTensor(1));
            weigthsTensor.setOrder(OrderType::RowMajor);

            auto weights = om.constant(weigthsTensor.getData(), {inputShape[0], inputShape[1], inputShape[2], 
                opIt->getOutputTensor(0)->getShape()[1]}, sourceTensor->getDType(), 
                sourceTensor->getOrder(), opIt->getName() + "_weights");

            auto conv2D = om.conv2D(sourceTensor, weights, {1, 1}, {0, 0, 0, 0});
            pass.log(Logger::MessageType::Info, "Replaced FullyConnected op " + opIt->getName() + " with " + conv2D->getName());

            if (opIt->hasAttr("bias"))
            {
                auto biasTensorName = opIt->get<std::string>("bias");
                om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
                pass.log(Logger::MessageType::Info, "Moved Bias attribute of FullyConnected op " + opIt->getName() + " to " + conv2D->getName());
            }

            for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                std::size_t inputIdx = sinkFlow->get<std::size_t>("sinkInput");
                sinkFlow.sink()->erase("input" + std::to_string(inputIdx));
                om.defineFlow(conv2D, sinkFlow.sink(), inputIdx); 
            }

            while(opIt.parentsSize() > 1)
            {
                auto paramOp = opIt.leftmostParent();
                ++paramOp;
                om.removeOp(paramOp);
            }
            
            om.removeOp(opIt);
            opIt = parentOpIt;

        }

    }

}