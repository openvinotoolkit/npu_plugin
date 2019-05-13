#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void standaloneActivationAsPostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void averageAsDepthWise(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(FullyConnectedAsConv2D)
        .setFunc(fullyConnectedAsConv2DFcn)
        .setDescription(
            "Replaces the fullyConnected op with conv2D using 1x1 kernels"
        );

        MV_REGISTER_PASS(StandaloneActivationAsPostOps)
        .setFunc(standaloneActivationAsPostOpsFcn)
        .setDescription(
            "Replaces unsupported standalone activation operations with identity operation + postOp activation"
        );

        MV_REGISTER_PASS(AverageAsDepthWise)
        .setFunc(averageAsDepthWise)
        .setDescription(
            "Replaces average Pooling Layer with a DeptwiseConvolution"
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

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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
            auto weightsData = opIt->getInputTensor(1)->getData();
            auto inputShape = sourceTensor->getShape();

            auto weights = om.constantDataElement(weightsData, {inputShape[mv::IO_WIDTH_DIMENSION], inputShape[mv::IO_HEIGHT_DIMENSION], inputShape[mv::IO_CHANNEL_DIMENSION],
                opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION]}, sourceTensor->getDType(),
                Order(Order::getRowMajorID(4)),{{},{},{},{}}, opIt->getName() + "_weights");

            auto conv2D = om.conv(sourceTensor, weights, {1, 1}, {0, 0, 0, 0}, 1);
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

void standaloneActivationAsPostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& targetDescriptor, mv::Element&, mv::json::Object&)
{
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType(opIt->getOpType());
        if(!targetDescriptor.opSupported(opType) && targetDescriptor.opSupportedAsPostOp(opType))
        {
            pass.log(Logger::MessageType::Debug, "Found " + opType + " - " + opIt->getName());

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            auto sourceTensor = parentOpIt->getOutputTensor(0);

            auto parentOpItType = parentOpIt->getOpType();

            Data::OpListIterator opToUse;

            //Input, Costant, Concat are not real operations, so we need to introduce an identity op
            if(parentOpItType == "Input" ||
               parentOpItType == "Costant" ||
               parentOpItType == "Concat")
            {
                sourceTensor = om.identity(sourceTensor);
                auto identityOp = om.getSourceOp(sourceTensor);
                opToUse = identityOp;
                pass.log(Logger::MessageType::Info, "Replaced " + opType + " with identity+postOp " + opToUse->getName());

            }
            else //no need for identity op, everything can be attached directly to previous op
            {
                opToUse = parentOpIt;
                pass.log(Logger::MessageType::Info, "Replaced " + opType + " by fusing it as a postOp to " + opToUse->getName());
            }

            opToUse->set("postOpType", opType);
            std::vector<std::string> attrKeys(opIt->attrsKeys());

            for(auto attrKey : attrKeys)
            {
                auto attrToSet = opIt->get(attrKey);
                opToUse->set(attrKey, attrToSet);
            }


            opIt = linkNewOperationsReplacement(parentOpIt, sourceTensor, om, opIt);
        }
    }
}

void averageAsDepthWise(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "AveragePool")
        {
            pass.log(Logger::MessageType::Debug, "Found AveragePool op " + opIt->getName());

            auto sourceTensor = opIt->getInputTensor(0);
            auto parentOpIt = om.getSourceOp(sourceTensor);

            auto inputShape = sourceTensor->getShape();
            std::array<unsigned short, 2> kSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            std::array<unsigned short, 2> stride = opIt->get<std::array<unsigned short, 2>>("stride");
            std::array<unsigned short, 4> padding = opIt->get<std::array<unsigned short, 4>>("padding");

            unsigned int total_shape = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * kSize[1] * kSize[0];
            double value = 1/double(kSize[0] * kSize[1]);

            unsigned short channel_multiplier = 1;

            auto name = opIt->getName();
            mv::Data::TensorIterator weights;
            if (sourceTensor->isDoubleType())
            {
                pass.log(Logger::MessageType::Debug, "Input tensor not quantized, generating non-quantized weights");
                std::vector<double> weightsData(total_shape, value);
                weights = om.constant(weightsData,
                                    {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                    sourceTensor->getDType(),
                                    Order(Order::getRowMajorID(4)));
            }
            else
            {
                pass.log(Logger::MessageType::Debug, "Input tensor quantized, generating quantized weights");
                // If the input model is quantized, then the replacement pass needs to create
                // quantization params for the weights parameter of the depthwise convolution.
                int64_t weightsVal = 1;
                std::vector<int64_t> weightsData(total_shape, weightsVal);

                std::vector<int64_t> zp = { 0 };
                std::vector<double> scale(1, value);
                std::vector<double> min = { 1 };
                std::vector<double> max = { 1 };
                mv::QuantizationParams weightsQuantParams(zp, scale, min, max);

                weights = om.constantInt(weightsData,
                                    {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                    sourceTensor->getDType(),
                                    Order(Order::getRowMajorID(4)),
                                    weightsQuantParams);
            }

            //Check the last argument name!!!
            mv::Data::TensorIterator depthwise_conv;
            if (sourceTensor->isQuantized())
            {
                pass.log(Logger::MessageType::Debug, "Passing quantization params from input to output");
                auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
                // use default dilation factor
                depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, quantParams, name + "_DepthwiseConv");
            }
            else
            {
                pass.log(Logger::MessageType::Debug, "No need for quantization params, since input is of a floating point type");
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, emptyQuantParams, name + "_DepthwiseConv");
            }

            auto depthwise_conv_op = om.getSourceOp(depthwise_conv);
            auto weightsOp = om.getSourceOp(weights);
            auto weights_tensor = depthwise_conv_op->getInputTensor(1);

            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                weightsOp->set<unsigned>("opId", currentOpId);
                depthwise_conv_op->set<unsigned>("opId", currentOpId);
            }
            pass.log(Logger::MessageType::Info, "Replaced AveragePool op " + opIt->getName() + " with " + depthwise_conv->getName());

            opIt = linkNewOperationsReplacement(parentOpIt, depthwise_conv, om, opIt);

        }
    }
}
