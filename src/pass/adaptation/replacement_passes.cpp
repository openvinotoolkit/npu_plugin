#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void fullyConnectedAsConv2DFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

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

void fullyConnectedAsConv2DFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::FullyConnected)
        {

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            auto sourceTensor = parentOpIt->getOutputTensor(0);

            auto weightsData = opIt->getInputTensor(1)->getData();
            auto inputShape = sourceTensor->getShape();

            Tensor weigthsTensor = *(opIt->getInputTensor(1));
            weigthsTensor.reorder(Order::RowMajor);

            auto weights = om.constant(weigthsTensor.getData(), mv::Shape(inputShape[0], inputShape[1], inputShape[2], 
                opIt->getOutputTensor(0)->getShape()[1]), sourceTensor->getDType(), 
                sourceTensor->getOrder(), opIt->getName() + "_weights");

            auto conv2D = om.conv2D(sourceTensor, weights, {1, 1}, {0, 0, 0, 0});
            if (opIt->hasAttr("bias"))
            {
                auto biasTensorName = opIt->getAttr("bias").getContent<std::string>();
                om.addAttr(om.getSourceOp(conv2D), "bias", Attribute(AttrType::StringType, biasTensorName));
            }

            for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                byte_type inputIdx = sinkFlow->getAttr("sinkInput").getContent<byte_type>();
                sinkFlow.sink()->removeAttr("input" + Printable::toString(inputIdx));
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