#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/base/exception/logic_error.hpp"
#include <cmath>

const size_t FULLY_CONNECTED_KERNEL = 1;

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void handleEltWiseDifferentScales(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void tensorsToFP16Fcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void tensorsToU8Fcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void interpAsAvgPoolingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void flattenAsReshapeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void scaleAsDepthwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replaceLargeAvgPoolFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);

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
            "Replaces Operations"
        );

        MV_REGISTER_PASS(EltwiseToSWEltwise)
        .setFunc(handleEltWiseDifferentScales)
        .setDescription(
            "Replaces Eltwise with SW Layer Eltwise in case scales of inputs are different"
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
    pass.log(mv::Logger::MessageType::Debug, "Replacement passes are starting");
    fullyConnectedAsConv2DFcn(pass, model);
    replaceLargeAvgPoolFcn(pass, model);
    //interpAsAvgPoolingFcn(pass, model); for now we are using SW layer
    averageAsDepthWiseFcn(pass, model);
    scaleAsDepthwiseFcn(pass, model);
    flattenAsReshapeFcn(pass, model);
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
        if (opIt->hasAttr("bias"))
        {
            auto biasTensorName = opIt->get<std::string>("bias");
            om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
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
void handleEltWiseDifferentScales(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto eltWiseOps = om.getOps("Eltwise");

    for (auto& opIt : eltWiseOps)
    {
        auto eltwiseType = opIt->get<std::string>("eltwiseType");
        if(eltwiseType == "Mult" || eltwiseType == "Divide")
            continue;

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
        auto scale1 = firstEltwiseInputTensorQuantizationParams.getScale();
        auto scale2 = secondEltwiseInputTensorQuantizationParams.getScale();

        auto size = scale1.size();
        std::vector <double> scaleDifference(size), absRelativeErrorScale(size), relativeErrorScale(size);
        std::transform(scale1.begin(),
                       scale1.end(),
                        scale2.begin(), scaleDifference.begin(), std::minus<double>());

        double (*fabs)(double) = &std::abs;
        std::transform(scaleDifference.begin(), scaleDifference.end(),
                       scale1.begin(), relativeErrorScale.begin(),
                       std::divides<double>());
        std::transform(relativeErrorScale.begin(),relativeErrorScale.end(),
                absRelativeErrorScale.begin(), fabs);
        for (auto it = absRelativeErrorScale.begin(); it != absRelativeErrorScale.end(); it++)
        {
            if (*it > 0.01)
                opIt->set<bool>("softwareExecuted", true);
        }
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
                auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
                avgPool = om.averagePool(sourceTensor, kSize, stride, {0,0,0,0}, false,"","floor",  mv::DType("Default"), quantParams, name + "_AvgPool");
            }
            else
            {
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                 avgPool = om.averagePool(sourceTensor, kSize, stride, {0,0,0,0}, false,"","floor",  mv::DType("Default"), emptyQuantParams, name + "_AvgPool");
            }

            auto avgOp = om.getSourceOp(avgPool);

            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                avgOp->set<unsigned>("opId", currentOpId);
            }
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

        if (opIt->hasAttr("bias"))
        {
            auto biasTensorName = opIt->get<std::string>("bias");
            om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
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
            weights = om.constant(weightsData,
                                {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(),
                                mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
        }
        else
        {
            int64_t weightsValue = 1;
            std::vector<int64_t> weightsData(total_shape, weightsValue);
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
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            // use default dilation factor
            depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, mv::DType("Default"), quantParams, name + "_DepthwiseConv");
        }
        else
        {
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

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto inputShape = sourceTensor->getShape();
        mv::QuantizationParams weightsTensorQuantizationParams = {{},{},{},{}};
        mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};

        auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        auto outputOrder = opIt->getOutputTensor(0)->getOrder();

        auto reshape = om.reshape(sourceTensor, outputShape, outputTensorType, outputTensorQuantizationParams,  opIt->getName() + "_reshape");

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

bool isPrime(unsigned short n) 
{ 
    // Check for existence of factor
    for(int i = 2; i < n; i++)
    {
        if (n % i == 0)
            return false;
    }
  
    return true; 
}

std::vector<std::pair<unsigned short, unsigned short>> getFactors(unsigned short n)
{
    std::vector<std::pair<unsigned short, unsigned short>> factors;
    for(int i = 2; i <= sqrt(n); i++)
    {
        if(n % i == 0)
        {
            // factors.push_back(std::make_pair(i, n/i)); // smaller, larger
	    factors.push_back(std::make_pair(n/i, i)); // larger, smaller
        }
    }
    return factors;
}

// Check for average pooling layers with kernels bigger than supported by hardware (11x11)
// and replace with equivalent two average pooling (approx equiv in case of prime kernel i.e. 13x13)
void replaceLargeAvgPoolFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
     MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto MAX_KERNEL = 11; // hardware limitation

    auto averagePoolOps = om.getOps("AveragePool");

    for (auto& opIt : averagePoolOps)
    {
        auto sourceTensor = opIt->getInputTensor(0);
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        auto parentOpIt = om.getSourceOp(sourceTensor);

        auto inputShape = sourceTensor->getShape();
        std::array<unsigned short, 2> kSize = opIt->get<std::array<unsigned short, 2>>("kSize");

        if((kSize[0] != kSize[1]) and (kSize[0] > MAX_KERNEL or kSize[1] > MAX_KERNEL))
        {
            // TODO deal with asymetric kernels too large
            continue;
        }

        // If average pool kernel size is greater than 11, we will turn it in to multiple depthwise convs here
        // Does the kernel size have two factors that are both less than 11? If so, use those kernel sizes and create
        // 2 new average pools to replace the existing average pool
        if(kSize[0] > MAX_KERNEL and !isPrime(kSize[0]))
        {
            std::vector<std::pair<unsigned short, unsigned short>> allFactors = getFactors(kSize[0]);
            std::pair<unsigned short, unsigned short> factors = allFactors.back(); // Get the most equal factors
 
            if ( factors.first > MAX_KERNEL or factors.second > MAX_KERNEL)
            {
                //TODO throw error, unable to split into appropriate size
                continue;
            } 

            // Calculate strides and padding based on new kernel sizes
            std::array<unsigned short, 2> stride0 = {factors.first, factors.first};
            std::array<unsigned short, 2> stride1 = {factors.second, factors.second};

	    // Padding relationship is (input size + pad ) / k = output size
            // pad = output*k - input
            double eachSize0 = (double) kSize[0]/factors.first;
            double paddedSize0 = ceil(eachSize0) * factors.first;
            unsigned short pad0 = paddedSize0 - inputShape[mv::IO_HEIGHT_DIMENSION];
            std::array<unsigned short, 4> padding0 = {0, pad0, 0, pad0};
            std::array<unsigned short, 4> padding1 = {0,0,0,0}; // Assumes output size is equal to second factor

            auto name = opIt->getName();
            mv::QuantizationParams quantParams({{}, {}, {}, {}});
            if (sourceTensor->isQuantized())
                quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            auto firstPool = om.averagePool(sourceTensor, {factors.first, factors.first}, stride0, 
                                                padding0, true, "", "floor", mv::DType("Default"), quantParams, name + "_AP0");
            auto firstPoolOp = om.getSourceOp(firstPool);

            unsigned currentOpId = 0;
            if(opIt->hasAttr("opId"))
            {
                currentOpId = opIt->get<unsigned>("opId");
                firstPoolOp->set<unsigned>("opId", currentOpId);
            }

            // First pool replaces original pool
            linkNewOperationsReplacement(parentOpIt, firstPool, om, opIt);

            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

            auto sourceFlowStart = firstPoolOp.leftmostOutput();

            for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
                flowsToRemove.push_back(sinkFlow);
            }
            // Remove old flow before creating second pool 
            for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
            {
                om.undefineFlow(flowsToRemove[flowIdx]);
            }

            // Second pool is new op to the model
            auto secondPool = om.averagePool(firstPool, {factors.second, factors.second}, stride1, 
                                                padding1, true, "", "floor", mv::DType("Default"), quantParams, name + "_AP1");
            auto secondPoolOp = om.getSourceOp(secondPool);
            secondPoolOp->set<unsigned>("opId", currentOpId);
        
            // Link rest of graph to new pool
            for(unsigned op = 0 ; op < opsToLink.size(); ++op)
            {
                opsToLink[op]->setInputTensor(secondPool, inputSlots[op], false);
                om.defineFlow(secondPool, opsToLink[op], inputSlots[op]);
            }
        }
        // Else the kernel size is a prime number and we will approximate kernels and use weights scaling
        // Note: approximate kernel sizes should be designed so the output tensor of the second depthwise
        // is the correct size for the network. The division scale of the weights will be used to improve accuracy.
        else if (kSize[0] > MAX_KERNEL and isPrime(kSize[0]))
        {
            // Use the factors for kernel size - 1, this is guaranteed to have factors, as it will be an even number > 11
            std::vector<std::pair<unsigned short, unsigned short>> allFactors = getFactors(kSize[0] - 1);
            std::pair<unsigned short, unsigned short> factors = allFactors.back(); // Get the most equal factors

            factors.second++; // These factors are for ksize - 1, so increase smaller factor by 1

            if ( factors.first > MAX_KERNEL or factors.second > MAX_KERNEL)
            {
                //TODO throw error, unable to split into appropriate size
                continue;
            }
            // Calculate strides and padding based on new kernel sizes
            std::array<unsigned short, 2> stride0 = {factors.first, factors.first};
            std::array<unsigned short, 2> stride1 = {factors.second, factors.second};

            double eachSize0 = (double) kSize[0]/factors.first;
            double paddedSize0 = ceil(eachSize0) * factors.first;
            unsigned short pad0 = paddedSize0 - inputShape[mv::IO_HEIGHT_DIMENSION];
            std::array<unsigned short, 4> padding0 = {0, pad0, 0, pad0};
            std::array<unsigned short, 4> padding1 = {0,0,0,0};

            unsigned int total_shape0 = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * factors.first * factors.first;

            unsigned short channel_multiplier = 1;

            auto name = opIt->getName();
            mv::Data::TensorIterator weights0;
            std::vector<int64_t> zp = { 0 };
            std::vector<double> min = { 1 };
            std::vector<double> max = { 1 };
            // Both depthwise will take 1/kernel_size as a scale (if was 1 dw would be kernel^2)
            double scaleValue = 1/double(kSize[0]);
            std::vector<double> scale(1, scaleValue);
            mv::QuantizationParams weightsQuantParams(zp, scale, min, max);

            // Create weights tensor
            if (sourceTensor->isDoubleType())
            {
                double weightsValue = 1;
                std::vector<double> weightsData(total_shape0, weightsValue);
                //NOTE: Weights have to be 1 and scale division, cause of better hardware accuracy
                weights0 = om.constant(weightsData,
                                {factors.first, factors.first, inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(), mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
            }
            else
            {
                int64_t weightsValue = 1;
                std::vector<int64_t> weightsData(total_shape0, weightsValue);
                // If the input model is quantized, then the replacement pass needs to create
                // quantization params for the weights parameter of the depthwise convolution.
                weights0 = om.constantInt(weightsData,
                                {factors.first, factors.first, inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(), mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
            }

            // Create first depthwise conv
            mv::Data::TensorIterator depthwise_conv0;
            if (sourceTensor->isQuantized())
            {
                auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
                // use default dilation factor
                depthwise_conv0 = om.depthwiseConv(sourceTensor, weights0, stride0, padding0, 1, mv::DType("Default"), quantParams, name + "_DepthwiseConv0");
            }
            else
            {
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                depthwise_conv0 = om.depthwiseConv(sourceTensor, weights0, stride0, padding0, 1, mv::DType("Default"), emptyQuantParams, name + "_DepthwiseConv0");
            }

            // Add first depthwise conv to op model
            auto depthwiseOp0 = om.getSourceOp(depthwise_conv0);
            auto weightsOp0 = om.getSourceOp(weights0);

            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                weightsOp0->set<unsigned>("opId", currentOpId);
                depthwiseOp0->set<unsigned>("opId", currentOpId);
            }
            depthwise_conv0->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            linkNewOperationsReplacement(parentOpIt, depthwise_conv0, om, opIt);

            // Remove old flow, remember to it to put next depthwise into model in correct place
            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

            auto sourceFlowStart = depthwiseOp0.leftmostOutput();

            for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
                flowsToRemove.push_back(sinkFlow);
            }
            // Remove old flow before creating new dw
            for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
            {
                om.undefineFlow(flowsToRemove[flowIdx]);
            }

            // Now generate the second depthwise conv, some parameters reused from above
            unsigned int total_shape1 = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * factors.second * factors.second;
            mv::Data::TensorIterator weights1;

            if (sourceTensor->isDoubleType())
            {
                double weightsValue = 1;
                std::vector<double> weightsData(total_shape1, weightsValue);
                //NOTE: Weights have to be 1 and scale division, cause of better hardware accuracy
                weights1 = om.constant(weightsData,
                                {factors.second, factors.second, inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(), mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
            }
            else
            {
                int64_t weightsValue = 1;
                std::vector<int64_t> weightsData(total_shape1, weightsValue);
                // If the input model is quantized, then the replacement pass needs to create
                // quantization params for the weights parameter of the depthwise convolution.
                weights1 = om.constantInt(weightsData,
                                {factors.second, factors.second, inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(), mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
            }

            mv::Data::TensorIterator depthwise_conv1;
            if (sourceTensor->isQuantized())
            {
                auto quantParams = depthwiseOp0->get<mv::QuantizationParams>("quantParams");
                depthwise_conv1 = om.depthwiseConv(depthwise_conv0, weights1, stride1, padding1, 1, mv::DType("Default"), quantParams, name + "_DepthwiseConv1");
            }
            else
            {
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                depthwise_conv1 = om.depthwiseConv(depthwise_conv0, weights1, stride1, padding1, 1, mv::DType("Default"), emptyQuantParams, name + "_DepthwiseConv1");
            }

            // Link up op model correctly
            auto depthwiseOp1 = om.getSourceOp(depthwise_conv1);
            auto weightsOp1 = om.getSourceOp(weights1);

            if(depthwiseOp0->hasAttr("opId"))
            {
                unsigned currentOpId = depthwiseOp0->get<unsigned>("opId");
                weightsOp1->set<unsigned>("opId", currentOpId);
                depthwiseOp1->set<unsigned>("opId", currentOpId);
            }
            depthwise_conv1->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

            for(unsigned op = 0 ; op < opsToLink.size(); ++op)
            {
                opsToLink[op]->setInputTensor(depthwise_conv1, inputSlots[op], false);
                om.defineFlow(depthwise_conv1, opsToLink[op], inputSlots[op]);
            }
        } // end primes
    } // end for avg pools
}
