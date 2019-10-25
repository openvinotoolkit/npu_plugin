#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"

static void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::DataModel dm, mv::OpModel om);
static void fuseReluFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fuseLeakyReluFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fusePowerFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fuseSigmoidFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fuseMinimumFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fuseMaximumFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fuseScaleFcn(mv::Data::OpListIterator &opIt, mv::DataModel dm, mv::OpModel om);
static void fuseSigmoidFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fuseBatchNormFcn(mv::Data::OpListIterator &opIt, mv::OpModel om);
static void fusePostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);


namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(FusePostOps)
        .setFunc(fusePostOpsFcn)
        .setDescription(
            "Fuses all the ops that will be converted to PPE Tasks and can be handled through hardware. "
            "Scale, Batch Norm from My-X\n"
        );
    }
}

void fusePostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    using namespace mv;

    OpModel om(model);
    DataModel dm(model);
    unsigned bias_nodes, sigmoid_nodes, relu_nodes, leakyRelu_nodes, power_nodes, minimum_nodes, maximum_nodes;
    bias_nodes = sigmoid_nodes = relu_nodes = leakyRelu_nodes = power_nodes = minimum_nodes = maximum_nodes = 0;
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Bias")
            bias_nodes++;
        else if (opIt->getOpType() == "Sigmoid")
            sigmoid_nodes++;
        else if (opIt->getOpType() == "Relu")
            relu_nodes++;
        else if (opIt->getOpType() == "LeakyRelu")
            leakyRelu_nodes++;
        else if (opIt->getOpType() == "Power")
            power_nodes++;
        else if (opIt->getOpType() == "MinimumDouble" || opIt->getOpType() == "MinimumInt")
            minimum_nodes++;
        else if (opIt->getOpType() == "MaximumDouble" || opIt->getOpType() == "MaximumInt")
            maximum_nodes++;
    }

    auto opIt = om.getInput();
    while (bias_nodes > 0 || sigmoid_nodes > 0 || relu_nodes > 0 || leakyRelu_nodes > 0
           || minimum_nodes > 0 || maximum_nodes > 0)
    {
        if (opIt->getOpType() == "Bias")
        {
            pass.log(Logger::MessageType::Debug, "Found Bias op " + opIt->getName());
            fuseBiasFcn(opIt, dm, om);
            bias_nodes--;
        }
        //REMAIN MY-X NOT POSTOP
//        else if (opIt->getOpType() == "Scale")
//        {
//            pass.log(Logger::MessageType::Debug, "Found Scale op " + opIt->getName());
//            fuseScaleFcn(opIt, dm, om);
//        }
        else if (opIt->getOpType() == "Sigmoid")
        {
            pass.log(Logger::MessageType::Debug, "Found Sigmoid op " + opIt->getName());
            fuseSigmoidFcn(opIt, om);
            sigmoid_nodes--;
        }
        else if (opIt->getOpType() == "Relu")
        {
            pass.log(Logger::MessageType::Debug, "Found Relu op " + opIt->getName());
            fuseReluFcn(opIt, om);
            relu_nodes--;
        }
        else if (opIt->getOpType() == "LeakyRelu")
        {
            pass.log(Logger::MessageType::Debug, "Found Leaky Relu op " + opIt->getName());
            fuseLeakyReluFcn(opIt, om);
            leakyRelu_nodes--;
        }
        else if (opIt->getOpType() == "LeakyRelu")
        {
            pass.log(Logger::MessageType::Debug, "Found Leaky Relu op " + opIt->getName());
            fuseLeakyReluFcn(opIt, om);
            leakyRelu_nodes--;
        }
        else if (opIt->getOpType() == "Power")
        {
            pass.log(Logger::MessageType::Debug, "Found Power op " + opIt->getName());
            fusePowerFcn(opIt, om);
            power_nodes--;
        }
        else if (opIt->getOpType() == "MinimumDouble" || opIt->getOpType() == "MinimumInt")
        {
            pass.log(Logger::MessageType::Debug, "Found Minimum op " + opIt->getName());
            fuseMinimumFcn(opIt, om);
            minimum_nodes--;
        }
        else if (opIt->getOpType() == "MaximumDouble" || opIt->getOpType() == "MaximumInt")
        {
            pass.log(Logger::MessageType::Debug, "Found Maximum op " + opIt->getName());
            fuseMaximumFcn(opIt, om);
            maximum_nodes--;
        }
        //REMAIN MY-X NOT POSTOP
//        else if (opIt->getOpType() == "BatchNormalization")
//        {
//            pass.log(Logger::MessageType::Debug, "Found Batch Norm op " + opIt->getName());
//            fuseBatchNormFcn(opIt, om);
//        }
        ++opIt;
    }

}

mv::Data::OpListIterator linkNewOperationsFuse(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt" || paramOp->getOpType() == "ConstantDataElement")
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::DataModel dm, mv::OpModel om)
{
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    if (parentOpIt->getOpType() == "Conv" ||
        parentOpIt->getOpType() == "FullyConnected" ||
        parentOpIt->getOpType() == "DepthwiseConv")
    {
        auto bias = *opIt->getInputTensor(1);
        auto biasOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        if (parentOpIt->hasAttr("bias"))
        {
            auto biasTensor = dm.getTensor(parentOpIt->get<std::string>("bias"));
            biasTensor->add(bias);
        }
        else
        {
            std::string biasTensorName = mv::createBiasName(parentOpIt->getName());
            mv::Data::TensorIterator biasTensor;
            if (bias.hasAttr("quantParams"))
                biasTensor = dm.defineTensor(mv::Tensor(biasTensorName, bias.getShape(), bias.getDType(), bias.getOrder(), bias.getData(), bias.get<mv::QuantizationParams>("quantParams")) );
            else
                biasTensor = dm.defineTensor(mv::Tensor(biasTensorName, bias.getShape(), bias.getDType(), bias.getOrder(), bias.getData()) );
            om.addAttr(parentOpIt, "bias", biasTensor->getName());
        }
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (biasOutputMemoryLocation.isForced())
        {
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", biasOutputMemoryLocation);
        }
    }
}

void fuseScaleFcn(mv::Data::OpListIterator &opIt, mv::DataModel dm, mv::OpModel om)
{
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    auto scaleOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    if (parentOpIt->getOpType() == "Conv")
    {
        auto scale = *opIt->getInputTensor(1);
        parentOpIt->getInputTensor(1)->multiply(scale);
        if (parentOpIt->hasAttr("bias"))
        {
            auto biasTensor = dm.getTensor(parentOpIt->get<std::string>("bias"));
            biasTensor->multiply(scale);
        }
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (scaleOutputMemoryLocation.isForced())
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", scaleOutputMemoryLocation);
    }
}

void fuseSigmoidFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto sigmoidOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::string>("postOpType", "Sigmoid");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (sigmoidOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", sigmoidOutputMemoryLocation);
}

void fuseMinimumFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto minimumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::vector<std::string>>("postOpTypes", {"Minimum"});
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    if (sourceTensor->getDType() == mv::DType("Float16"))
        parentOpIt->set<double>("minimum", opIt->get<double>("minimum"));
    else
        parentOpIt->set<int64_t>("minimum", opIt->get<int64_t>("minimum"));

    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (minimumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", minimumOutputMemoryLocation);
}

void fuseMaximumFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto maximumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    std::vector<std::string> postOpTypes = {};
    if (parentOpIt->hasAttr("postOpTypes"))
        postOpTypes = parentOpIt->get<std::vector<std::string>>("postOpTypes");

    postOpTypes.push_back("Maximum");
    parentOpIt->set<std::vector<std::string>>("postOpTypes", postOpTypes);
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    if (sourceTensor->getDType() == mv::DType("Float16"))
        parentOpIt->set<double>("maximum", opIt->get<double>("maximum"));
    else
        parentOpIt->set<int64_t>("maximum", opIt->get<int64_t>("maximum"));

    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (maximumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", maximumOutputMemoryLocation);
}

void fuseReluFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto reluOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::string>("postOpType", "Relu");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (reluOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", reluOutputMemoryLocation);
}

void fuseLeakyReluFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto reluOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::string>("postOpType", "LeakyRelu");
    parentOpIt->set<double>("alpha", opIt->get<double>("alpha"));
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (reluOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", reluOutputMemoryLocation);
}

void fusePowerFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto powerOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::string>("postOpType", "Power");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (powerOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", powerOutputMemoryLocation);
}

void fuseBatchNormFcn(mv::Data::OpListIterator &opIt, mv::OpModel om)
{
    auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto batchNormName = opIt->getName();
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    auto bnMean = *opIt->getInputTensor(1);
    auto bnVar = *opIt->getInputTensor(2);
    auto bnOffset = *opIt->getInputTensor(3);
    auto bnScale = *opIt->getInputTensor(4);
    double bnEps = opIt->get<double>("eps");
    auto scaleParam = mv::math::divide(bnScale, mv::math::sqrt(mv::math::add(bnVar, bnEps)));
    auto offsetParam = mv::math::subtract(bnOffset, mv::math::multiply(bnMean, scaleParam));
    auto offset = om.constantDataElement(offsetParam.getData(), offsetParam.getShape(), offsetParam.getDType(),
        offsetParam.getOrder(),{{},{},{},{}}, batchNormName + "_offset");

    mv::Data::TensorIterator sourceTensor;

    if (bnMean.getShape().ndims() == 1)
    {
        if (parentOpIt->getOpType() == "Conv")
        {
            parentOpIt->getInputTensor(1)->multiply(scaleParam);
            sourceTensor = parentOpIt->getOutputTensor(0);
        }
        else
        {
            auto scale = om.constantDataElement(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
            sourceTensor = om.scale(opIt->getInputTensor(0), scale);
            parentOpIt = om.getSourceOp(sourceTensor);
        }
    }
    else
    {
        auto scale = om.constantDataElement(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
        sourceTensor = om.multiply({opIt->getInputTensor(0), scale});
        parentOpIt = om.getSourceOp(sourceTensor);
    }

    if (offsetParam.getShape().ndims() == 1)
        sourceTensor = om.bias(sourceTensor, offset);
    else
        sourceTensor = om.add({sourceTensor, offset});
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (outputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
}
