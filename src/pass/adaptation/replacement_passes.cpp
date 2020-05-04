#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <cmath>

const size_t FULLY_CONNECTED_KERNEL = 1;

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void handleEltWiseDifferentScales(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void decideTasksPrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void interpAsAvgPoolingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void flattenAsReshapeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void topKAsArgMaxFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void scaleAsDepthwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void placeNeutralMaxPoolBefore(mv::OpModel om, mv::Data::OpListIterator task);
void replaceLargeAvgPoolFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replacePoolReshapePatternFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);

namespace mv
{

    namespace pass
    {

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

        MV_REGISTER_PASS(DecideTasksPrecision)
        .setFunc(decideTasksPrecisionFcn)
        .setDescription(
            "Replaces DPU Tasks with no Quant Params with float DPU Tasks"
        );
    }

}

void placeNeutralMaxPoolBefore(mv::OpModel om, mv::Data::OpListIterator task)
{
    auto inputFlow = task.leftmostInput();
    auto neutralMaxPool = om.maxPool(task->getInputTensor(0), {1,1}, {1,1}, {0, 0, 0, 0},
                                     false, mv::DType("Float16"), {{0}, {1.0f}, {}, {}}, task->getName() + "MaxPool");
    auto maxPoolOp = om.getSourceOp(neutralMaxPool);
    maxPoolOp->set<unsigned>("opId", task->get<unsigned>("opId"));
    maxPoolOp->set<bool>("softwareExecuted", true);
    om.undefineFlow(inputFlow);
    task->setInputTensor(neutralMaxPool, 0, false);
    om.defineFlow(neutralMaxPool, task, 0);
}


void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                       mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    fullyConnectedAsConv2DFcn(pass, model);
    replacePoolReshapePatternFcn(pass, model);
    replaceLargeAvgPoolFcn(pass, model);
    topKAsArgMaxFcn(pass, model);
    //interpAsAvgPoolingFcn(pass, model); for now we are using SW layer
    averageAsDepthWiseFcn(pass, model);
    scaleAsDepthwiseFcn(pass, model);
    flattenAsReshapeFcn(pass, model);
}

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
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
void handleEltWiseDifferentScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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

void decideTasksPrecisionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    //Note: This pass will be dis-abled and enabled only for cases that we want to execute fp16 precision dpu tasks
    //these tasks are marked cause they have no quant params...Important here is that the Z-major Convolution has
    //the limitation of sparse tensors, so we need to have a maxpool(neutral) before that
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    std::vector<std::string> futere_dpu_types = {"Eltwise", "DepthwiseConv","MaxPool"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType
            = om.getOpsOfTypes(futere_dpu_types);
    mv::QuantizationParams emptyQuantizationParams = {{}, {}, {}, {}};
    mv::QuantizationParams neutralQuantizationParams = {{0}, {1.0}, {}, {}};
    std::vector<mv::Data::OpListIterator> simpleDpuCases = operationsOfType["Eltwise"];
    std::merge(operationsOfType["DepthwiseConv"].begin(), operationsOfType["DepthwiseConv"].end(),
            operationsOfType["MaxPool"].begin(), operationsOfType["MaxPool"].end(), simpleDpuCases.begin());

    for (auto& opIt : simpleDpuCases)
    {
        if(opIt->get<bool>("softwareExecuted"))
            continue;

        if (opIt->get<mv::QuantizationParams>("quantParams").isEmpty())
            opIt->set<bool>("softwareExecuted", true);
        else
        {
            if (opIt->get<mv::QuantizationParams>("quantParams").isNeutral())
                opIt->set<bool>("softwareExecuted", true);
        }
    }

    auto convOps = om.getOps("Conv");
    for (auto& opIt : convOps)
    {
        if (opIt->get<mv::QuantizationParams>("quantParams").isEmpty())
            opIt->set<bool>("softwareExecuted", true);
        else
        {
            if (opIt->get<mv::QuantizationParams>("quantParams").isNeutral())
                opIt->set<bool>("softwareExecuted", true);
        }
        if (opIt->hasAttr("softwareExecuted"))
        {
             if (opIt->get<bool>("softwareExecuted"))
                placeNeutralMaxPoolBefore(om, opIt);
        }
    }
}

void interpAsAvgPoolingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
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
            auto factor = (unsigned short) (inHeight / outHeight);
            auto parentOpIt = om.getSourceOp(sourceTensor);

            std::array<unsigned short, 2> kSize({factor, factor});
            std::array<unsigned short, 2> stride({factor, factor});
            auto name = opIt->getName();

            //Check the last argument name!!!
            mv::Data::TensorIterator avgPool;
            if (sourceTensor->isQuantized())
            {
                auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
                avgPool = om.averagePool(sourceTensor, kSize, stride, {0,0,0,0}, false,  mv::DType("Default"), quantParams, name + "_AvgPool");
            }
            else
            {
                mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
                 avgPool = om.averagePool(sourceTensor, kSize, stride, {0,0,0,0}, false,  mv::DType("Default"), emptyQuantParams, name + "_AvgPool");
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

void scaleAsDepthwiseFcn(const mv::pass::PassEntry& , mv::ComputationModel& model)
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
        convOp->set<bool>("isScaleShift", true);
    }
}

void averageAsDepthWiseFcn(const mv::pass::PassEntry& , mv::ComputationModel& model)
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
        mv::QuantizationParams emptyWeightsQuantParams = {{},{},{},{}};

        if (sourceTensor->isDoubleType())
        {
            double weightsValue = scaleValue;
            std::vector<double> weightsData(total_shape, weightsValue);
            //NOTE: For FP, weights quant params not used - put divisor in weights directly instead of scale
            weights = om.constant(weightsData,
                                {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(),
                                mv::Order(mv::Order::getRowMajorID(4)), emptyWeightsQuantParams);
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

void topKAsArgMaxFcn(const mv::pass::PassEntry& , mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    auto topKOps = om.getOps("TopK");

    for (auto& opIt : topKOps)
    {
        //Check if first output has no data flows
        //check if mode and sort are different from what is supported by argmax
        auto firstoutput = opIt->getOutputTensor(0);
        auto attrs = opIt->getAttrs();

        if(firstoutput->hasAttr("flows") ||
            (attrs.at("mode").get<std::string>().compare("max") != 0) ||
            (attrs.at("sort").get<std::string>().compare("index") != 0))
            throw ArgumentError("topKAsArgMaxFcn", "flows", "", "cannot convert topK to argMax");// TODO replace with continue when we add topK support

        auto outputMemoryLocation = opIt->getOutputTensor(1)->get<mv::Tensor::MemoryLocation>("Location");

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto inputShape = sourceTensor->getShape();


        auto dtype = attrs.at("dType").get<mv::DType>();
        auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
        auto out_max_val = 0; //only support this for this conversion
        auto top_k = attrs.at("top_k").get<int64_t>();
        auto axis = attrs.at("axis").get<int64_t>();


        auto argmax = om.argmax(sourceTensor, out_max_val, top_k, axis, dtype, quantParams, opIt->getName() + "_argmax");

        auto argmaxOp = om.getSourceOp(argmax);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            argmaxOp->set<unsigned>("opId", currentOpId);
        }
        linkNewOperationsReplacement(parentOpIt, argmax, om, opIt);
        argmax->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

void flattenAsReshapeFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
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

std::vector<std::pair<unsigned short, unsigned short>> getFactorsList(unsigned short n)
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

std::pair<unsigned short, unsigned short> getFactors(unsigned short kernelSize)
{
    std::vector<std::pair<unsigned short, unsigned short>> allFactors = getFactorsList(kernelSize);
    std::pair<unsigned short, unsigned short> factors;
    if (allFactors.empty()) // Original kernel size IS PRIME
    {
        // Use the factors for kernel size - 1, this is guaranteed to have factors, as it will be an even number > 11
        allFactors = getFactorsList(kernelSize - 1);
        factors = allFactors.back(); // Get the most equal factors

        // These factors are for ksize - 1, so increase smaller factor by 1
        if( factors.first == factors.second)
            factors.first++;
        else
            factors.second++;
    }
    else // Original kernel size NOT PRIME
    {
        factors = allFactors.back(); // Get the most equal factors
    }
    return factors;
}

unsigned short getPad(std::pair<unsigned short, unsigned short> factors, size_t inputShape, size_t outputShape)
{
    double updatedOutput = outputShape * factors.second;
    unsigned short pad = updatedOutput*factors.first -  inputShape;

    return pad;
}

mv::Data::TensorIterator createPartialDepthwise(mv::OpModel om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceTensor,
                                                 std::string name, unsigned short originalKernel, std::array<unsigned short, 2> newKernel,
                                                 std::array<unsigned short, 4> padding)
{
    auto inputShape = sourceTensor->getShape();

    // Calculate strides based on new kernel sizes
    std::array<unsigned short, 2> stride = {newKernel[0], newKernel[1]};

    unsigned int total_shape = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * newKernel[0] * newKernel[1];

    unsigned short channel_multiplier = 1;

    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };
    // Both depthwise will take 1/original_kernel_size as a scale (if was 1 dw would be kernel^2)
    // Note: For non-prime kernels, could take scale of each exactly replacing ap/dw,
	// but using original kernel for scale improves observed accuracy
    double scaleValue = 1/double(originalKernel);
    std::vector<double> scale(1, scaleValue);
    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    mv::QuantizationParams emptyWeightsQuantParams = {{},{},{},{}};


    // Create weights tensor
    if (sourceTensor->isDoubleType())
	{
		double weightsValue = scaleValue;
		std::vector<double> weightsData(total_shape, weightsValue);
		//NOTE: For FP, weights quant params not used - put divisor in weights directly instead of scale
		weights = om.constant(weightsData,
                                {newKernel[0], newKernel[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(), mv::Order(mv::Order::getRowMajorID(4)), emptyWeightsQuantParams);
    }
	else
    {
		int64_t weightsValue = 1;
		std::vector<int64_t> weightsData(total_shape, weightsValue);
		// If the input model is quantized, then the replacement pass needs to create
		// quantization params for the weights parameter of the depthwise convolution.
		weights = om.constantInt(weightsData,
                                {newKernel[0], newKernel[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                sourceTensor->getDType(), mv::Order(mv::Order::getRowMajorID(4)), weightsQuantParams);
	}
    // Create depthwise conv
	mv::Data::TensorIterator depthwise_conv;
	if (sourceTensor->isQuantized())
	{
        auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
        // use default dilation factor
        depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, mv::DType("Default"), quantParams, name);
	}
	else
	{
        mv::QuantizationParams emptyQuantParams({{}, {}, {}, {}});
        depthwise_conv = om.depthwiseConv(sourceTensor, weights, stride, padding, 1, mv::DType("Default"), emptyQuantParams, name);
 	}

    // Add depthwise conv to op model
	auto depthwiseOp = om.getSourceOp(depthwise_conv);
	auto weightsOp = om.getSourceOp(weights);

	if(opIt->hasAttr("opId"))
	{
        unsigned currentOpId = opIt->get<unsigned>("opId");
        weightsOp->set<unsigned>("opId", currentOpId);
        depthwiseOp->set<unsigned>("opId", currentOpId);
	}
    auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
	depthwise_conv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

    return depthwise_conv;
}

bool matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto opIt = it;

    for (auto& layer : pattern) {
        if (opIt->getOpType() != layer) {
            return false;
        }

        opIt = om.getSourceOp(opIt->getInputTensor(0));
    }

    return true;
}

bool canReplaceAveragePool(mv::Data::OpListIterator first, mv::Data::OpListIterator second, mv::OpModel& om) {
    auto first_attrs = first->getAttrs({"opId"});
    auto second_attrs = second->getAttrs({"opId"});

    if (!(first_attrs["quantParams"].get<mv::QuantizationParams>().getScale() == second_attrs["quantParams"].get<mv::QuantizationParams>().getScale() &&
        first_attrs.at("stride") == second_attrs.at("stride") &&
        first_attrs.at("padding") == first_attrs.at("padding") &&
        first_attrs.at("exclude_pad") == second_attrs.at("exclude_pad") &&
        first_attrs.at("dType") == second_attrs.at("dType")))
        return false;

    auto first_kernel = first_attrs["kSize"].get<std::array<unsigned short, 2>>();
    auto second_kernel = second_attrs["kSize"].get<std::array<unsigned short, 2>>();

    auto reshape_dims = om.getSourceOp(second->getInputTensor(0))->get<mv::Shape>("shape");

    // This condition handles these cases
    // nx1 -> reshape -> reshape -> nx1
    // nx1 -> reshape -> 1xn
    return (first_kernel[0] == second_kernel[1] && first_kernel[1] == 1 && second_kernel[0] == 1) ||
            (first_kernel == second_kernel && first_kernel[0] == 1 && reshape_dims[0] == 1);
}

void replacePoolReshapePatternFcn(const mv::pass::PassEntry& , mv::ComputationModel& model) {
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    // Note: Pattern is reversed. First AveragePool in vector is the last AveragePool in graph
    //const std::vector<std::string> pattern = {"AveragePool", "Reshape", "Reshape", "AveragePool", "Reshape"};
    const std::vector<std::string> pattern = {"Reshape", "AveragePool", "Reshape", "Reshape", "AveragePool"};
    auto ops = om.getOps(*pattern.begin());

    auto can_replace_pool = [&pattern, &om](mv::Data::OpListIterator opIt) -> bool {
        std::vector<mv::Data::OpListIterator> poolOps;

        auto it = opIt;
        for (size_t i = 0; i < pattern.size(); ++i) {
            if (it->getOpType() == "AveragePool") {
                poolOps.push_back(it);
            }

            it = om.getSourceOp(it->getInputTensor(0));
        }
        assert(poolOps.size() == 2);

        return canReplaceAveragePool(poolOps[1], poolOps[0], om);
    };

    for (auto& opIt : ops)
    {
        if (!opIt) {
            continue;
        }

        if (matchPattern(pattern, opIt, model) && can_replace_pool(opIt)) {
            auto poolingOp = om.getSourceOp(opIt->getInputTensor(0));
            auto kernel = std::max(poolingOp->get<std::array<unsigned short, 2>>("kSize")[0], poolingOp->get<std::array<unsigned short, 2>>("kSize")[1]);
            std::array<unsigned short, 2> kSize = {kernel, kernel};
            auto op_attrs =  poolingOp->getAttrs({"kSize"});

            auto it = om.getSourceOp(opIt->getInputTensor(0));

            for (size_t i = 0; i < pattern.size() - 1; ++i) {
                auto parentOpIt = om.getSourceOp( it->getInputTensor(0));
                auto sourceTensor = parentOpIt->getOutputTensor(0);
                it = linkNewOperationsRemove(parentOpIt, sourceTensor, om, it);
            }

            auto ap = om.averagePool(it->getOutputTensor(0),
                    kSize,
                    op_attrs.at("stride"),
                    op_attrs.at("padding"),
                    op_attrs.at("exclude_pad"),
                    op_attrs.at("dType"),
                    op_attrs.at("quantParams"));

            if(opIt->hasAttr("opId")) {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                ap->set<unsigned>("opId", currentOpId);
                om.getSourceOp(ap)->set<unsigned>("opId", currentOpId);
            }

            linkNewOperationsReplacement(it, ap, om, opIt);
        }
    }
}

// Check for average pooling layers with kernels bigger than supported by hardware (11x11)
// and replace with equivalent two average pooling (approx equiv in case of prime kernel i.e. 13x13)
// Example: 13x13 kernel is replaced with 2 depthwise convolutions, each 4x4 kernel, stride 4, scale 1/13
// Example: 14x14 kernel is replaced with 1 depthwise 7x7 kernel, stride 7, scale 1/14 followed by
// depthwise 2x2 kernel, stride 2, scale 1/14
void replaceLargeAvgPoolFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
     MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto MAX_KERNEL = 11; // hardware limitation

    auto averagePoolOps = om.getOps("AveragePool");

    for (auto& opIt : averagePoolOps)
    {
        std::array<unsigned short, 2> kSize = opIt->get<std::array<unsigned short, 2>>("kSize");

        if(kSize[0] < MAX_KERNEL and kSize[1] < MAX_KERNEL) // can do as single depthwise, skip
            continue;

        auto kernelSize = kSize[0];
        auto largeDim = 0;
        auto asymmetricCase = false;
        auto asymmetricBothKernelsLarge = false;

        if((kSize[0] != kSize[1]) and (kSize[0] > MAX_KERNEL or kSize[1] > MAX_KERNEL))
        {
            if (kSize[0] > MAX_KERNEL and kSize[1] > MAX_KERNEL)
                asymmetricBothKernelsLarge = true;

            // deal with asymetric kernels when one dim is larger than MAX_KERNEL
            asymmetricCase = true;
            if (kSize[0] <  kSize[1])
            {
                kernelSize = kSize[1];
                largeDim = 1;
            }
        }

        pass.log(mv::Logger::MessageType::Debug, "largeKernel " +  std::to_string(kernelSize) + " kernelDim " + std::to_string(largeDim));
        auto name = opIt->getName();
        auto sourceTensor = opIt->getInputTensor(0);

        auto parentOpIt = om.getSourceOp(sourceTensor);

        auto inputShape = sourceTensor->getShape();
        auto outputShape = opIt->getOutputTensor()[0]->getShape();

        std::vector<std::pair<unsigned short, unsigned short>> allFactors;
        std::pair<unsigned short, unsigned short> factors;
        std::pair<unsigned short, unsigned short> factorsDim2;

        // If average pool kernel size is greater than 11, we will turn it in to multiple depthwise convs here
        // Note: Kernel sizes should be chosen so the output tensor of the second depthwise
        // is the correct size for the network. The division scale of the weights will be used to improve accuracy.

        factors = getFactors(kernelSize);
        if (factors.first > MAX_KERNEL or factors.second > MAX_KERNEL)
        {
            //TODO throw error, unable to split into appropriate size
            std::cout << "ERROR: factors are larger the MAX_KERNEL " << std::endl;
            continue;
        }

        pass.log(mv::Logger::MessageType::Debug, "factors " +  std::to_string(factors.first) + " , " + std::to_string(factors.second));

        if (asymmetricBothKernelsLarge)
        {
            factorsDim2 = getFactors(kSize[1-largeDim]);

            if (factorsDim2.first > MAX_KERNEL or factorsDim2.second > MAX_KERNEL)
            {
                //TODO throw error, unable to split into appropriate size
                continue;
            }
            pass.log(mv::Logger::MessageType::Debug, "factors2 " +  std::to_string(factorsDim2.first) + " , " + std::to_string(factorsDim2.second));
        }

        // Padding relationship is (input size + pad) / k = output size
        unsigned short pad = getPad(factors, inputShape[largeDim], outputShape[largeDim]);

        std::array<unsigned short, 4> padding = {0, pad, 0, pad};
        std::array<unsigned short, 2> newKernel = {factors.first, factors.first};

        if (asymmetricCase)
        {
            if (asymmetricBothKernelsLarge)
            {
                unsigned short pad2 = getPad(factorsDim2, inputShape[1-largeDim], outputShape[1-largeDim]);
                if (largeDim == 0)
                {
                    padding[3] = pad2;
                    newKernel[1] = factorsDim2.first;
                }
                else
                {
                    padding[1] = pad2;
                    newKernel[0] = factorsDim2.first;
                }
            }
            else
            {
                if (largeDim == 0)
                {
                    padding[3] = 0;
                    newKernel[1] = kSize[1];
                }
                else
                {
                    padding[1] = 0;
                    newKernel[0] = kSize[0];
                }
            }
        }

        mv::Data::TensorIterator depthwise_conv0 = createPartialDepthwise(om, opIt, sourceTensor, name + "_DepthwiseConv0",
                                                                            kernelSize, newKernel, padding);

        linkNewOperationsReplacement(parentOpIt, depthwise_conv0, om, opIt);

        // Remove old flow, remember to it to put next depthwise into model in correct place
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

        auto depthwiseOp0 = om.getSourceOp(depthwise_conv0);
        auto sourceFlowStart = depthwiseOp0.leftmostOutput();

        if (asymmetricCase)
        {
            depthwiseOp0->set<unsigned>("asymmetricKernel", 1-largeDim);//record dimension we need workload to stride over.
        }

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

        newKernel[0] = newKernel[1] = factors.second;
        auto scaleVal = kSize[0];
        if (asymmetricCase)
        {
            if (asymmetricBothKernelsLarge)
            {
                if (largeDim == 0)
                {
                    scaleVal = kSize[1];
                    newKernel[1] = factorsDim2.second;
                }
                else
                {
                    scaleVal = kSize[0];
                    newKernel[0] = factorsDim2.second;
                }
            }
            else
            {
                if (largeDim == 0)
                {
                    scaleVal = kSize[1];
                    newKernel[1] =  outputShape[1]/depthwise_conv0->getShape()[1];
                }
                else
                {
                    scaleVal = kSize[0];
                    newKernel[0] = outputShape[0]/depthwise_conv0->getShape()[0];
                }
            }

        }
        pass.log(mv::Logger::MessageType::Debug, "newKernel " +  std::to_string(newKernel[0]) + " , " + std::to_string(newKernel[1]));

        // Now generate the second depthwise conv
        mv::Data::TensorIterator depthwise_conv1 = createPartialDepthwise(om, depthwiseOp0, depthwise_conv0,
                                                                        name + "_DepthwiseConv1", scaleVal, newKernel, {0,0,0,0});

        for(unsigned op = 0 ; op < opsToLink.size(); ++op)
        {
            opsToLink[op]->setInputTensor(depthwise_conv1, inputSlots[op], false);
            om.defineFlow(depthwise_conv1, opsToLink[op], inputSlots[op]);
        }
    } // end for
}
