#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void standaloneActivationAsPostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populatedTensorsToFP16Fcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void unpopulatedTensorsToFP16Fcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);


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
        .setFunc(averageAsDepthWiseFcn)
        .setDescription(
            "Replaces average Pooling Layer with a DeptwiseConvolution"
        );

        MV_REGISTER_PASS(PopulatedTensorsToFP16)
        .setFunc(populatedTensorsToFP16Fcn)
        .setDescription(
            "Replaces full precision populated tensors with FP16 populated tensors"
        );

        MV_REGISTER_PASS(UnpopulatedTensorsToFP16)
        .setFunc(unpopulatedTensorsToFP16Fcn)
        .setDescription(
            "Replaces full precision populated tensors dtype"
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

void populatedTensorsToFP16Fcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);

    auto kernelOp = om.opBegin();
    while(kernelOp != om.opEnd())
    {
        if(kernelOp.outputsSize() > 0)
        {
            auto outputTensor = kernelOp->getOutputTensor(0);
            auto originalDTypeSize = outputTensor->getDType().getSizeInBits();
            if(outputTensor->isPopulated() && (originalDTypeSize == 64 || originalDTypeSize == 32))
            {
                auto opId = kernelOp->get<unsigned>("opId");

                std::vector<double> oldData = kernelOp->getOutputTensor(0)->getDoubleData();
                std::vector<int64_t> newData(oldData.size());

                for(unsigned i = 0; i < oldData.size(); ++i)
                    newData[i] = mv::fp32_to_fp16(oldData[i]);

                auto kernelShape = kernelOp->getOutputTensor(0)->getShape();
                auto kernelOrder = kernelOp->getOutputTensor(0)->getOrder();

                auto backup = kernelOp;
                ++kernelOp;
                auto outputDataFlows = mv::getOutputDataFlow(om, backup);
                auto newKernel = om.constantInt(newData, kernelShape, mv::DType("Float16"), kernelOrder);
                auto newKernelOp = om.getSourceOp(newKernel);
                newKernelOp->set<unsigned>("opId", opId);

                mv::setOutputDataFlow(om, newKernel, outputDataFlows);
            }
            else
                ++kernelOp;
        }
        else
            ++kernelOp;
    }
}

void unpopulatedTensorsToFP16Fcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);

    for(auto tensorIt = om.tensorBegin(); tensorIt != om.tensorEnd(); ++tensorIt)
    {
        auto originalDTypeSize = tensorIt->getDType().getSizeInBits();
        if(originalDTypeSize == 64 || originalDTypeSize == 32)
            tensorIt->setDType(mv::DType("Float16"));
    }
}

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "FullyConnected")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            pass.log(Logger::MessageType::Debug, "Found FullyConnected op " + opIt->getName());

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            auto sourceTensor = parentOpIt->getOutputTensor(0);
            auto weightsData = opIt->getInputTensor(1)->getData();
            auto inputShape = sourceTensor->getShape();
            mv::QuantizationParams weightsTensorQuantizationParams = {{},{},{},{}};
            mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};

            if (opIt->getInputTensor(1)->isQuantized())
            {
                weightsTensorQuantizationParams = opIt->getInputTensor(1)->get<mv::QuantizationParams>("quantParams");
                outputTensorQuantizationParams = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
            }
            auto weights = om.constantDataElement(weightsData, {inputShape[mv::IO_WIDTH_DIMENSION], inputShape[mv::IO_HEIGHT_DIMENSION], inputShape[mv::IO_CHANNEL_DIMENSION],
            opIt->getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION]}, sourceTensor->getDType(),
            mv::Order::getZMajorID(4),weightsTensorQuantizationParams, opIt->getName() + "_weights");

            auto conv2D = om.conv(sourceTensor, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"), outputTensorQuantizationParams);
            pass.log(Logger::MessageType::Info, "Replaced FullyConnected op " + opIt->getName() + " with " + conv2D->getName());

            if (opIt->hasAttr("bias"))
            {
                auto biasTensorName = opIt->get<std::string>("bias");
                om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
                pass.log(Logger::MessageType::Info, "Moved Bias attribute of FullyConnected op " + opIt->getName() + " to " + conv2D->getName());
            }

            auto convOp = om.getSourceOp(conv2D);
            auto weightsOp = om.getSourceOp(weights);

            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                weightsOp->set<unsigned>("opId", currentOpId);
                convOp->set<unsigned>("opId", currentOpId);
            }

            opIt = linkNewOperationsReplacement(parentOpIt, conv2D, om, opIt);
            conv2D->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        }

    }
}

void standaloneActivationAsPostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& targetDescriptor, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType(opIt->getOpType());
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

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
            sourceTensor->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        }
    }
}

void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "AveragePool")
        {
            pass.log(Logger::MessageType::Debug, "Found AveragePool op " + opIt->getName());

            auto sourceTensor = opIt->getInputTensor(0);
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

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
                depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, mv::DType("Default"), quantParams, name + "_DepthwiseConv");
            }
            else
            {
                pass.log(Logger::MessageType::Debug, "No need for quantization params, since input is of a floating point type");
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, mv::DType("Default"), emptyQuantParams, name + "_DepthwiseConv");
            }

            auto depthwiseConvOp = om.getSourceOp(depthwise_conv);
            auto weightsOp = om.getSourceOp(weights);

            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                weightsOp->set<unsigned>("opId", currentOpId);
                depthwiseConvOp->set<unsigned>("opId", currentOpId);
            }
            pass.log(Logger::MessageType::Info, "Replaced AveragePool op " + opIt->getName() + " with " + depthwise_conv->getName());
            depthwise_conv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            opIt = linkNewOperationsReplacement(parentOpIt, depthwise_conv, om, opIt);

        }
    }
}
