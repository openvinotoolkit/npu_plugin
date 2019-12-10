#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"

const size_t FULLY_CONNECTED_KERNEL = 1;

static void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void handleEltWiseDifferentScales(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void tensorsToFP16Fcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void tensorsToU8Fcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void interpAsAvgPoolingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void flattenAsReshapeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void scaleAsDepthwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void removeIdentityOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TensorsToFP16)
        .setFunc(tensorsToFP16Fcn)
        .setDescription(
            "Replaces full precision tensors with FP16 tensors"
        );

        MV_REGISTER_PASS(TensorsToU8)
        .setFunc(tensorsToU8Fcn)
        .setDescription(
            "Replaces quantized int8 tensors with U8 tensors"
        );

        MV_REGISTER_PASS(ReplacementOps)
        .setFunc(replacementOpsFcn)
        .setDescription(
            "Replaces Average with Depthwise and FullyConnected with Convolution"
        );

    }

}


mv::Data::OpListIterator linkNewOperationsReplacement(mv::Data::OpListIterator parentOpIt,
                            mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt"
                || paramOp->getOpType() == "ConstantDataElement")
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

    if(sourceTensor == om.tensorEnd())
        sourceTensor = parentOpIt->getOutputTensor(0);

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}


void tensorsToFP16Fcn(const mv::pass::PassEntry&  , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    using namespace mv;
    OpModel om(model);

    auto kernelOp = om.getInput();
    while (kernelOp != om.opEnd())
    {
        if(kernelOp.outputsSize() > 0)
        {
            auto outputTensor = kernelOp->getOutputTensor(0);
            if(outputTensor->get<mv::DType>("dType") == mv::DType("Float64") ||
               outputTensor->get<mv::DType>("dType") == mv::DType("Float32"))
            {
                auto opId = kernelOp->get<unsigned>("opId");
                if (outputTensor->isPopulated())
                {
                    std::vector<double> oldData = kernelOp->getOutputTensor(0)->getDoubleData();
                    std::vector<int64_t> newData(oldData.size());
                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    if(outputTensor->hasAttr("quantParams"))
                        quantParams = outputTensor->get<mv::QuantizationParams>("quantParams");

                    for(unsigned i = 0; i < oldData.size(); ++i)
                        newData[i] = mv::fp32_to_fp16(oldData[i]);
                    auto kernelShape = kernelOp->getOutputTensor(0)->getShape();
                    auto kernelOrder = kernelOp->getOutputTensor(0)->getOrder();
                    //with data flows I am finding where the op was attached to attache the new one!!!
                    auto outputDataFlows = mv::getOutputDataFlow(om, kernelOp);

                    auto newKernel = om.constantInt(newData, kernelShape, mv::DType("Float16"), kernelOrder, quantParams);
                    auto newKernelOp = om.getSourceOp(newKernel);
                    newKernelOp->set<unsigned>("opId", opId);
                    newKernelOp->set<mv::DType>("dType",  mv::DType("Float16"));
                    mv::setOutputDataFlow(om, newKernel, outputDataFlows);
                }
                else
                {
                    mv::DType newType = mv::DType("Float16");
                    outputTensor->setDType(newType);
                    kernelOp->set<mv::DType>("dType",  mv::DType("Float16"));
                    ++kernelOp;
                }
            }
            else
                ++kernelOp;
        }
        else
            ++kernelOp;
    }
}

// Pass logic:
// Runtime will handle the input, we uniform all the rest to UInt8
void tensorsToU8Fcn(const mv::pass::PassEntry&  , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);

    int64_t zeroPointShift = 128;
    auto sourceDType = mv::DType("Int8");
    auto targetDType = mv::DType("UInt8");

    auto kernelOp = om.getInput();
    auto inputType = kernelOp->getOutputTensor(0)->getDType();
    if(inputType == mv::DType("Int8"))
        throw std::runtime_error("Compiler doesn't support I8 inputs for the moment, please rescale your data to U8");

    for (; kernelOp != om.opEnd(); ++kernelOp)
    {
        if(kernelOp.outputsSize() > 0)
        {
            auto outputTensor = kernelOp->getOutputTensor(0);
            auto outputTensorDType = outputTensor->get<mv::DType>("dType");
            if(outputTensorDType == sourceDType)
            {
                mv::DType newType = targetDType;
                auto quantParams = outputTensor->get<mv::QuantizationParams>("quantParams");
                auto quantParamsZp = quantParams.getZeroPoint();
                for(auto& zp: quantParamsZp)
                    zp += zeroPointShift;
                quantParams = mv::QuantizationParams(quantParamsZp, quantParams.getScale(),{},{});
                outputTensor->setDType(newType);
                kernelOp->set<mv::DType>("dType",  newType);
                outputTensor->set<mv::QuantizationParams>("quantParams", quantParams);
                kernelOp->set<mv::QuantizationParams>("quantParams", quantParams);
                if (outputTensor->isPopulated())
                    for(unsigned i = 0; i < outputTensor->size(); ++i)
                        outputTensor->at(i) += zeroPointShift;
            }
        }
    }
}

void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                       mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    removeIdentityOps(pass, model);
    fullyConnectedAsConv2DFcn(pass, model);
    //interpAsAvgPoolingFcn(pass, model); for now we are using SW layer
    averageAsDepthWiseFcn(pass, model);
    scaleAsDepthwiseFcn(pass, model);
    handleEltWiseDifferentScales(pass, model);
    flattenAsReshapeFcn(pass, model);
}

void removeIdentityOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto identityOps = om.getOps("Identity");

    for (auto& opIt : identityOps)
    {

        pass.log(mv::Logger::MessageType::Debug, "Found Identity op " + opIt->getName());

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);

        linkNewOperationsReplacement(parentOpIt, om.tensorEnd(), om, opIt);
    }
}

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    auto fullyConnectedOps = om.getOps("FullyConnected");

    for (auto& opIt : fullyConnectedOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        pass.log(Logger::MessageType::Debug, "Found FullyConnected op " + opIt->getName());

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto weightsData = opIt->getInputTensor(1)->getData();
        auto inputShape = sourceTensor->getShape();
        mv::QuantizationParams weightsTensorQuantizationParams = {{},{},{},{}};
        mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};

        if (opIt->getInputTensor(1)->isQuantized())
        {
            weightsTensorQuantizationParams = opIt->getInputTensor(1)->get<mv::QuantizationParams>("quantParams");
            outputTensorQuantizationParams = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        }
        auto weights = om.constantDataElement(weightsData, {FULLY_CONNECTED_KERNEL, FULLY_CONNECTED_KERNEL, inputShape[mv::IO_CHANNEL_DIMENSION],
        opIt->getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION]}, sourceTensor->getDType(),
        mv::Order::getZMajorID(4), weightsTensorQuantizationParams, opIt->getName() + "_weights");
        auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");

        auto conv2D = om.conv(sourceTensor, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, outputTensorType, outputTensorQuantizationParams,  opIt->getName() + "_2DConv");
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

        linkNewOperationsReplacement(parentOpIt, conv2D, om, opIt);
        conv2D->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

//NOTE: This pass will handle cases that we have Convs -> Eltwise for testing ResNet first of all....
//General solution dequantize the input Tensors of these special Elwise, even with sw de-quantize
void handleEltWiseDifferentScales(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto eltWiseOps = om.getOps("Eltwise");

    for (auto& opIt : eltWiseOps)
    {
        auto eltwiseType = opIt->get<std::string>("eltwiseType");
        if(eltwiseType == "Mult" || eltwiseType == "Divide")
            continue;
        pass.log(mv::Logger::MessageType::Debug, "Found Eltwise op " + opIt->getName());

        auto firstEltwiseInputTensor = opIt->getInputTensor(0);
        auto secondEltwiseInputTensor = opIt->getInputTensor(1);

        mv::QuantizationParams firstEltwiseInputTensorQuantizationParams = {{}, {}, {}, {}};
        mv::QuantizationParams secondEltwiseInputTensorQuantizationParams = {{}, {}, {}, {}};

        if (firstEltwiseInputTensor->isQuantized())
            firstEltwiseInputTensorQuantizationParams =
                    firstEltwiseInputTensor->get<mv::QuantizationParams>("quantParams");
        if (secondEltwiseInputTensor->isQuantized())
            secondEltwiseInputTensorQuantizationParams =
                    secondEltwiseInputTensor->get<mv::QuantizationParams>("quantParams");

        if (firstEltwiseInputTensorQuantizationParams.getScale() !=
                secondEltwiseInputTensorQuantizationParams.getScale())
            opIt->set<bool>("softwareExecuted", true);

    }
}

void interpAsAvgPoolingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto interpOps = om.getOps("Interp");

    for (auto& opIt : interpOps)
    {
        pass.log(mv::Logger::MessageType::Debug, "Found Interp op " + opIt->getName());

        auto sourceTensor = opIt->getInputTensor(0);
        auto inputShape = sourceTensor->getShape();
        auto outputTensor = opIt->getOutputTensor(0);
        auto outputShape = outputTensor->getShape();

        auto inWidth = inputShape[mv::IO_WIDTH_DIMENSION];
        auto inHeight = inputShape[mv::IO_HEIGHT_DIMENSION];
        auto outWidth = outputShape[mv::IO_WIDTH_DIMENSION];
        auto outHeight = outputShape[mv::IO_HEIGHT_DIMENSION];
        if (inWidth > outWidth && inHeight > outHeight &&
             (inHeight % outHeight == 0) && (inWidth % outWidth == 0) &&
              (inHeight / outHeight) == inWidth / outWidth)
        {
            pass.log(mv::Logger::MessageType::Debug, "Found Interp op that will be replaced with AvgPooling " + opIt->getName());

            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto factor = inHeight / outHeight;
            auto parentOpIt = om.getSourceOp(sourceTensor);

            std::array<unsigned short, 2> kSize({factor, factor});
            std::array<unsigned short, 2> stride({factor, factor});
            auto name = opIt->getName();

            //Check the last argument name!!!
            mv::Data::TensorIterator avgPool;
            if (sourceTensor->isQuantized())
            {
                pass.log(mv::Logger::MessageType::Debug, "Passing quantization params from input to output");
                auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
                avgPool = om.averagePool(sourceTensor, kSize, stride, {0,0,0,0}, false,"","floor",  mv::DType("Default"), quantParams, name + "_AvgPool");
            }
            else
            {
                pass.log(mv::Logger::MessageType::Debug, "No need for quantization params, since input is of a floating point type");
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                 avgPool = om.averagePool(sourceTensor, kSize, stride, {0,0,0,0}, false,"","floor",  mv::DType("Default"), emptyQuantParams, name + "_AvgPool");
            }

            auto avgOp = om.getSourceOp(avgPool);

            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                avgOp->set<unsigned>("opId", currentOpId);
            }
            pass.log(mv::Logger::MessageType::Info, "Replaced Interp op " + opIt->getName() + " with " + avgOp->getName());
            avgOp->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            linkNewOperationsReplacement(parentOpIt, avgPool, om, opIt);
        }
    }
}

void scaleAsDepthwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto scaleOps = om.getOps("Scale");

    for (auto& opIt : scaleOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        pass.log(mv::Logger::MessageType::Debug, "Found FullyConnected op " + opIt->getName());

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto weightsData = opIt->getInputTensor(1)->getData();
        auto inputShape = sourceTensor->getShape();
        mv::QuantizationParams weightsTensorQuantizationParams = {{},{},{},{}};
        mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};

        if (opIt->getInputTensor(1)->isQuantized())
        {
            weightsTensorQuantizationParams = opIt->getInputTensor(1)->get<mv::QuantizationParams>("quantParams");
            outputTensorQuantizationParams = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        }

        if (parentOpIt->getOpType() == "Conv")
            continue;

        auto weights = om.constantDataElement(weightsData, {FULLY_CONNECTED_KERNEL, FULLY_CONNECTED_KERNEL, inputShape[mv::IO_CHANNEL_DIMENSION],
        1}, sourceTensor->getDType(),
        mv::Order::getZMajorID(4), weightsTensorQuantizationParams, opIt->getName() + "_weights");
        auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");

        auto conv2D = om.depthwiseConv(sourceTensor, weights, {1, 1}, {0, 0, 0, 0}, 1, outputTensorType, outputTensorQuantizationParams,  opIt->getName() + "_DepthwiseConv");
        pass.log(mv::Logger::MessageType::Info, "Replaced scale op " + opIt->getName() + " with " + conv2D->getName());

        if (opIt->hasAttr("bias"))
        {
            auto biasTensorName = opIt->get<std::string>("bias");
            om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
            pass.log(mv::Logger::MessageType::Info, "Moved Bias attribute of scale op " + opIt->getName() + " to " + conv2D->getName());
        }

        auto convOp = om.getSourceOp(conv2D);
        auto weightsOp = om.getSourceOp(weights);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            weightsOp->set<unsigned>("opId", currentOpId);
            convOp->set<unsigned>("opId", currentOpId);
        }

        linkNewOperationsReplacement(parentOpIt, conv2D, om, opIt);
        conv2D->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto averagePoolOps = om.getOps("AveragePool");

    for (auto& opIt : averagePoolOps)
    {
        pass.log(mv::Logger::MessageType::Debug, "Found AveragePool op " + opIt->getName());

        auto sourceTensor = opIt->getInputTensor(0);
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        auto parentOpIt = om.getSourceOp(sourceTensor);

        auto inputShape = sourceTensor->getShape();
        std::array<unsigned short, 2> kSize = opIt->get<std::array<unsigned short, 2>>("kSize");
        std::array<unsigned short, 2> stride = opIt->get<std::array<unsigned short, 2>>("stride");
        std::array<unsigned short, 4> padding = opIt->get<std::array<unsigned short, 4>>("padding");

        unsigned int total_shape = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * kSize[1] * kSize[0];
        double scaleValue = 1/double(kSize[0] * kSize[1]);

        unsigned short channel_multiplier = 1;

        auto name = opIt->getName();
        mv::Data::TensorIterator weights;
        std::vector<int64_t> zp = { 0 };
        std::vector<double> min = { 1 };
        std::vector<double> max = { 1 };

        std::vector<double> scale(1, scaleValue);
        mv::QuantizationParams weightsQuantParams(zp, scale, min, max);

        if (sourceTensor->isDoubleType())
        {
            double weightsValue = 1;
            std::vector<double> weightsData(total_shape, weightsValue);
            //NOTE: Weights have to be 1 and scale division, cause of better hardware accuracy
            pass.log(mv::Logger::MessageType::Debug, "Input tensor not quantized, generating non-quantized weights");
            weights = om.constant(weightsData,
                                {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(),
                                mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
        }
        else
        {
            int64_t weightsValue = 1;
            std::vector<int64_t> weightsData(total_shape, weightsValue);
            pass.log(mv::Logger::MessageType::Debug, "Input tensor quantized, generating quantized weights");
            // If the input model is quantized, then the replacement pass needs to create
            // quantization params for the weights parameter of the depthwise convolution.
            weights = om.constantInt(weightsData,
                                {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(),
                                mv::Order(mv::Order::getRowMajorID(4)),
                                weightsQuantParams);
        }

        //Check the last argument name!!!
        mv::Data::TensorIterator depthwise_conv;
        if (sourceTensor->isQuantized())
        {
            pass.log(mv::Logger::MessageType::Debug, "Passing quantization params from input to output");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            // use default dilation factor
            depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, mv::DType("Default"), quantParams, name + "_DepthwiseConv");
        }
        else
        {
            pass.log(mv::Logger::MessageType::Debug, "No need for quantization params, since input is of a floating point type");
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
        pass.log(mv::Logger::MessageType::Info, "Replaced AveragePool op " + opIt->getName() + " with " + depthwise_conv->getName());
        depthwise_conv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        linkNewOperationsReplacement(parentOpIt, depthwise_conv, om, opIt);
    }
}

void flattenAsReshapeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    auto flattenOps = om.getOps("Flatten");

    for (auto& opIt : flattenOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        pass.log(Logger::MessageType::Debug, "Found Flatten op " + opIt->getName());

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto inputShape = sourceTensor->getShape();
        mv::QuantizationParams weightsTensorQuantizationParams = {{},{},{},{}};
        mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};

        auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        auto outputOrder = opIt->getOutputTensor(0)->getOrder();

        auto reshape = om.reshape(sourceTensor, outputShape, outputOrder.toString(), outputTensorType, outputTensorQuantizationParams,  opIt->getName() + "_reshape");
        pass.log(Logger::MessageType::Info, "Replaced Flatten op " + opIt->getName() + " with " + reshape->getName());

        auto reshapeOp = om.getSourceOp(reshape);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            reshapeOp->set<unsigned>("opId", currentOpId);
        }
        linkNewOperationsReplacement(parentOpIt, reshape, om, opIt);
        reshape->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}
