#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>
#include "include/mcm/pass/pass_utils.hpp"

static void computeTensorsQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateOutputQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void postTrainingQuantize(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignConcatScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void placeReQuantizeDepthwiseBefore(mv::OpModel om, mv::Data::OpListIterator concat, mv::Data::TensorIterator inputTensor, std::size_t index, double &weightScale, double &alignedScale, int64_t &alignedZeroPoint);
//static void compensateDepthWiseAfter(mv::OpModel om, mv::Data::OpListIterator nextOp, mv::Data::OpListIterator concat);
//static std::vector<mv::Data::OpListIterator> findNextConcat(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor);

template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ComputeTensorsQuantParams)
        .setFunc(computeTensorsQuantParams)
        .setDescription(
            "This pass computes the appropriate quantize params extends and prepares them for serialization."
        );

        MV_REGISTER_PASS(PostTrainingQuantize)
        .setFunc(postTrainingQuantize)
        .setDescription(
            "The pass will estimate output tensor quantization param where quantization is needed."
        );
    }
}

void postTrainingQuantize(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                       mv::TargetDescriptor& td, mv::Element& e0, mv::Element& e1)
{
    alignConcatScales(pass, model, td, e0, e1);
    updateOutputQuantParams(pass, model, td, e0, e1);
}

void updateOutputQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    //NOTE: This pass will generate output Quantization Params when they are not defined...
    //Here we search for the minimum, maximum possible solution (better range) for the output Activation Tensor
    //Input(Imin,Imax)     Weights(Wmin,Wmax)
    // \                   /
    //  \                 /
    //   \               /
    //    \             /
    //     \           /
    //      \         /
    //       \       /
    //          Conv
    //           |
    //       Output(Omin,Omax)
    //           |
    //        Bias(Bmin,Bmax)
    // Suggestion: Omin = Imin * Wmin * kernel_w * kernel_h * input_channels, Rmin = Omin + Bmin
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::vector<std::string> convolution_types = {"Conv", "DepthwiseConv", "ChannelMajorConvolution"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfConvolution = om.getOpsOfTypes(convolution_types);
    std::vector <mv::Data::OpListIterator> convolutions = {};
    convolutions.reserve(operationsOfConvolution["Conv"].size() + operationsOfConvolution["Depthwise"].size() + operationsOfConvolution["ChannelMajorConvolution"].size());
    convolutions.insert(convolutions.end(), operationsOfConvolution["Conv"].begin(), operationsOfConvolution["Conv"].end());
    double inf = std::numeric_limits<double>::infinity();
    auto maxPoolOps = om.getOps("MaxPool");
    for(auto& opIt : maxPoolOps)
    {
        auto output = opIt->getOutputTensor(0);
        auto input = opIt->getInputTensor(0);

        if (!output->hasAttr("quantParams")
                || output->get<mv::QuantizationParams>("quantParams").isNeutral())
        {
            if (!input->hasAttr("quantParams"))
            {
                if (input->get<mv::QuantizationParams>("quantParams").isNeutral())
                    continue;
            }
            else
            {
                auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");

                output->set<mv::QuantizationParams>("quantParams", inputQuantization);
                opIt->set<mv::QuantizationParams>("quantParams", inputQuantization);
            }
        }

    }
    for(auto& opIt : convolutions)
    {
        auto output = opIt->getOutputTensor(0);
        auto input = opIt->getInputTensor(0);
        auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

        if (!output->hasAttr("quantParams")
                || output->get<mv::QuantizationParams>("quantParams").isNeutral())
        {
            double outputMin = inf;
            double outputMax = -inf;

            std::vector<double> outMin(outputChannels, inf);
            std::vector<double> outMax(outputChannels, -inf);

            //Note: if input Tensor has min, max of infs...we need to compute them
            updateInfMinMaxPerTensor(input);

            auto& newInputQuantization = input->get<mv::QuantizationParams>("quantParams");
            auto weights = opIt->getInputTensor("weights");
            auto kernelShape = weights->getShape();
            auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
            auto weights_scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
            auto weights_zp = extendToK(outputChannels, weightsQuantization.getZeroPoint(), weights->getName());

            //input/output quantization are per tensor, weights, bias quantization are per channel
            std::vector<double> outScale(1);
            std::vector<int64_t> outZp(1);
            auto minIn = newInputQuantization.getMin();
            auto maxIn = newInputQuantization.getMax();

            bool hasBias = opIt->hasAttr("bias");
            mv::Data::TensorIterator bias;
            if (hasBias)
            {
                bias = dm.getTensor(opIt->get<std::string>("bias"));
            }
            double_t real_weight, real_bias;
            for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
            {
                double sum_weight = 0;
                double outputMinC = 0;
                double outputMaxC = 0;
                double biasScale = weights_scale[k] * newInputQuantization.getScale()[0];

                for (size_t c = 0; c < kernelShape[mv::KERNEL_INPUT_CHANNELS]; c++)
                    for (size_t h = 0; h < kernelShape[mv::KERNEL_HEIGHT]; h++)
                        for (size_t w = 0; w < kernelShape[mv::KERNEL_WIDTH]; w++)
                        {
                            auto currWeight = (int64_t)weights->at({w,h,c,k});
                            real_weight = ((int64_t) currWeight - weights_zp[k]) * weights_scale[k];

                            sum_weight += real_weight;
                        }

                outputMaxC = maxIn[0] * sum_weight;
                outputMinC = minIn[0] * sum_weight;
//                if (outputMinC > outputMaxC)
//                //could happen if weight is negative
//                {
//                    auto temp = outputMaxC;
//                    outputMaxC = outputMinC;
//                    outputMinC = temp;
//                }
                if (hasBias)
                {
                    real_bias = ((int64_t) bias->at(k)) * biasScale;
                    outputMinC += real_bias;
                    outputMaxC += real_bias;
                }

                if (opIt->hasAttr("leakyAlpha"))
                {
                    auto alpha = opIt->get<double>("leakyAlpha");
                    if (outputMinC < 0)
                        outputMinC = outputMinC*alpha;
                }

                outMax[k] = outputMaxC;
                outMin[k] = outputMinC;
            }
            outputMin = *std::min_element(outMin.begin(), outMin.end());
            outputMax = *std::max_element(outMax.begin(), outMax.end());

            calcZeroPointAndScalePerTensor(outputMax, outputMin, outScale[0], outZp[0]);

            mv::QuantizationParams newOutputQuantization = {outZp,outScale,{outputMin},{outputMax}};
            output->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
            opIt->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
        }
    }
}

void placeReQuantizeDepthwiseBefore(mv::OpModel om, mv::Data::OpListIterator concat, mv::Data::TensorIterator inputTensor, std::size_t index, double &weightScale, double &alignedScale, int64_t &alignedZeroPoint)
{
    //FIND THE APPROPRIATE FLOW
    auto inputFlow = concat.leftmostInput();
    while(inputFlow != om.flowEnd())
    {
        auto tensor = inputFlow->getTensor();
        if (tensor->getName() == inputTensor->getName())
        {
            break;
        }
        ++inputFlow;
    }
    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };

    std::vector<double> scale(1, weightScale);
    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    int64_t weightsValue = 1;
    std::vector<int64_t> weightsData(inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
    weights = om.constantInt(weightsData,
                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order(mv::Order::getRowMajorID(4)),
                        weightsQuantParams);
    auto reQuantizeDepthwise = om.depthwiseConv(inputTensor, weights, {1,1}, {0, 0, 0, 0},
                        1, mv::DType("UInt8"), {{alignedZeroPoint},{alignedScale},{},{}}, concat->getName() + inputTensor->getName() + "Depthwise" + std::to_string(index));
    reQuantizeDepthwise->set<double>("oldScale", inputTensor->get<double>("oldScale"));
    auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
    auto weightsOp = om.getSourceOp(weights);
    reQuantizeDepthwiseOp->set<unsigned>("opId", concat->get<unsigned>("opId"));
    weightsOp->set<unsigned>("opId", concat->get<unsigned>("opId"));
    om.undefineFlow(inputFlow);
    concat->setInputTensor(reQuantizeDepthwise, index, false);
    om.defineFlow(reQuantizeDepthwise, concat, index);
}

//void compensateDepthWiseAfter(mv::OpModel om, mv::Data::OpListIterator nextOp, mv::Data::OpListIterator concat)
//{
//    auto inputFlow = nextOp.leftmostInput();
//    mv::Data::TensorIterator weights;
//    std::vector<int64_t> zp = { 0 }; // why always assume ZP 0?? it is in this case but doesnt have to be always
//    std::vector<double> min = { 1 };
//    std::vector<double> max = { 1 };

//    std::vector<double> scale(concat->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION], 1 );
//    double masterScale = concat->getInputTensor()[0]->get<mv::QuantizationParams>("quantParams").getScale()[0];
//    int64_t weightsValue = 1;
//    std::vector<int64_t> weightsData(concat->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
//    std::size_t sumChannels = 0;
//    for (std::size_t i = 0; i < concat->getInputTensor().size(); i++)
//    {
//        auto oldScale = concat->getInputTensor()[i]->get<double>("oldScale");
//        for (std::size_t outputChannel = sumChannels;  outputChannel < sumChannels + concat->getInputTensor()[i]->getShape()[mv::IO_CHANNEL_DIMENSION];
//             outputChannel++)
//        {
//            scale[outputChannel] = oldScale/masterScale;
//        }
//        sumChannels +=concat->getInputTensor()[i]->getShape()[mv::IO_CHANNEL_DIMENSION];
//    }

//    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
//    auto inputTensor = concat->getOutputTensor()[0];
//    weights = om.constantInt(weightsData,
//                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
//                        mv::DType("UInt8"),
//                        mv::Order(mv::Order::getRowMajorID(4)),
//                        weightsQuantParams);

//    auto compensateDepthWise = om.depthwiseConv(inputTensor, weights, {1,1}, {0, 0, 0, 0},
//                1, mv::DType("UInt8"), {{concat->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").getZeroPoint()[0]},{masterScale},{},{}}, nextOp->getName() + "compensateDepthwise");
//    auto compensateDepthWiseOp = om.getSourceOp(compensateDepthWise);
//    auto weightsOp = om.getSourceOp(weights);
//    compensateDepthWiseOp->set<unsigned>("opId", nextOp->get<unsigned>("opId"));
//    weightsOp->set<unsigned>("opId", nextOp->get<unsigned>("opId"));
//    om.undefineFlow(inputFlow);
//    nextOp->setInputTensor(compensateDepthWise, 0, false);
//    om.defineFlow(compensateDepthWise, nextOp, 0);
//}

static std::vector<mv::Data::OpListIterator> findNextConcat(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
    for(auto flowName : flowsNames)
    {
        auto df = dataModel.getDataFlow(flowName);
        sinkOperations.push_back(df.sink());
    }
    return sinkOperations;
}

static void markCompensatedConcats(std::vector<mv::Data::OpListIterator> &concats)
{
    //NOTE: Mark the concats that need compensation
    for(auto& concatIt : concats)
    {
        auto tempQuant =  concatIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams");
        auto tempScale = concatIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale()[0];
        concatIt->set<bool>("compensateNeed", false);
        concatIt->getInputTensor(0)->set<double>("oldScale", tempScale);

        for (std::size_t i = 1; i < concatIt->getInputTensor().size(); i++)
        {
            auto concatInputQuantParams = concatIt->getInputTensor(i)->get<mv::QuantizationParams>("quantParams");
            //NOTE: Activation tensors need to support only one value
            if (std::abs(tempScale - concatInputQuantParams.getScale()[0])/tempScale >
                    0.01)
            {
                auto oldScale = concatInputQuantParams.getScale()[0];
                concatIt->getInputTensor(i)->set<double>("oldScale", oldScale);
                concatIt->set<bool>("compensateNeed", true);
            }
            else
                continue;
        }
    }
}

static mv::QuantizationParams computeAlignedQuantParams(mv::Data::OpListIterator &concatIt)
{
    std::vector<double> minInputFloats, maxInputFloats = {};

    //NOTE: Compute the min/max of every tensor that goes in the Concat
    for (std::size_t i = 0; i < concatIt->getInputTensor().size(); i++)
    {
        //Note: if input Tensor has min, max of infs...we need to compute them
        updateInfMinMaxPerTensor(concatIt->getInputTensor(i));

        auto& inputQuantization = concatIt->getInputTensor(i)->get<mv::QuantizationParams>("quantParams");

        minInputFloats.push_back(inputQuantization.getMin()[0]);
        maxInputFloats.push_back(inputQuantization.getMax()[0]);
    }

    double minConcatScale = *std::min_element(minInputFloats.begin(), minInputFloats.end());
    double maxConcatScale = *std::max_element(maxInputFloats.begin(), maxInputFloats.end());

    double masterScale = 1.0;
    int64_t zeroPoint = 0;
    calcZeroPointAndScalePerTensor(maxConcatScale, minConcatScale, masterScale, zeroPoint);

    return {{zeroPoint},{masterScale},{minConcatScale},{maxConcatScale}};
}

void alignConcatScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto concats = om.getOps("Concat");

    // NOTE: For concats that go to concats the solution need to be recursive
    // Covering 2 recursion rounds now
    std::vector<mv::Data::OpListIterator> childConcats = {};
    for(auto& concatIt : concats)
    {
        auto nextOp = findNextConcat(dm, concatIt->getOutputTensor()[0])[0];
        if (nextOp->getOpType() == "Concat")
            childConcats.push_back(nextOp);
    }

    for (auto& childConcatIt : childConcats)
        concats.erase(std::remove(concats.begin(), concats.end(), childConcatIt), concats.end());

    markCompensatedConcats(concats);
    for(auto& concatIt : concats)
    {
        if (concatIt->get<bool>("compensateNeed"))
        {
            mv::QuantizationParams masterQuant = computeAlignedQuantParams(concatIt);
            for (std::size_t i = 0; i < concatIt->getInputTensor().size(); i++)
            {
                if (std::abs(masterQuant.getScale()[0] - concatIt->getInputTensor(i)->get<mv::QuantizationParams>("quantParams").getScale()[0])/masterQuant.getScale()[0] <= 0.01)
                    continue;
                double weightScale = 1.0f;
               
                placeReQuantizeDepthwiseBefore(om, concatIt, concatIt->getInputTensor(i), i, weightScale, masterQuant.getScale()[0], masterQuant.getZeroPoint()[0]);
            }
            concatIt->getOutputTensor(0)->set<mv::QuantizationParams>("quantParams", masterQuant);
            concatIt->set<mv::QuantizationParams>("quantParams", masterQuant);
        }
    }
    markCompensatedConcats(childConcats);
    for(auto& concatIt : childConcats)
    {
        if (concatIt->get<bool>("compensateNeed"))
        {
            mv::QuantizationParams masterQuant = computeAlignedQuantParams(concatIt);
            for (std::size_t i = 0; i < concatIt->getInputTensor().size(); i++)
            {
                if (masterQuant.getScale()[0] == concatIt->getInputTensor()[i]->get<mv::QuantizationParams>("quantParams").getScale()[0])
                    continue;
                double weightScale = 1.0f;
                placeReQuantizeDepthwiseBefore(om, concatIt, concatIt->getInputTensor(i), i, weightScale, masterQuant.getScale()[0], masterQuant.getZeroPoint()[0]);
            }
            concatIt->getOutputTensor(0)->set<mv::QuantizationParams>("quantParams", masterQuant);
            concatIt->set<mv::QuantizationParams>("quantParams", masterQuant);
        }
    }
}

void computeTensorsQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto dpuTasks = om.getOps("DPUTask");

    for(auto& opIt : dpuTasks)
    {
         std::string taskOp = opIt->get<std::string>("taskOp");
         bool isEltwise = taskOp == "Eltwise";
         bool isEltwiseMult = false;
         bool isEltwiseAddSub = false;
         if(isEltwise)
         {
             auto eltwiseType = opIt->get<std::string>("eltwiseType");
             if(eltwiseType == "Add" || eltwiseType == "Subtract")
                 isEltwiseAddSub = true;
             if(eltwiseType == "Multiply")
                 isEltwiseMult = true;
         }
         bool isConv = (taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "ChannelMajorConvolution");
         if (isConv || taskOp == "MaxPool" || isEltwiseMult || isEltwiseAddSub)
         {
            auto output = opIt->getOutputTensor(0);
            auto input = opIt->getInputTensor(0);
            auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
            outputChannels = mv::round_up(outputChannels, 16);

            std::vector<int> shift(outputChannels, 0);
            std::vector<int16_t> mScaled(outputChannels, 0);

            if (output->hasAttr("quantParams") && input->hasAttr("quantParams") &&
             output->isQuantized() && input->isQuantized())
            {
                 // Quantization for Gemmlowp output
                 // S1 = weight scale
                 // S2 = input activation scale
                 // S3 = output activation scale
                 // m  = (S1 * S2)/S3, scale for MAC output
                 // zeroPointScaled = output zero point scaled to MAC output precision
                 // biasScaled = bias scaled to MAC output precision

                 auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                 //inputQuantization.extendParamsToOutputChannelSize(outputChannels);

                 auto scale = extendToK(outputChannels, inputQuantization.getScale(), input->getName());
                 std::vector<float> S2(scale.begin(), scale.end());

                 mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                 scale = extendToK(outputChannels, outputQuantization.getScale(), output->getName());
                 std::vector<float> S3(scale.begin(), scale.end());

                 auto zeroPointU =  extendToK(outputChannels, outputQuantization.getZeroPoint(), output->getName());
                 std::vector<int32_t> zeroPoint(zeroPointU.begin(), zeroPointU.end());

                 bool isPooling = taskOp == "MaxPool";
                 //Workaround for HW bug #227
                 if (isPooling)
                 {
                     auto inZP = extendToK(outputChannels, inputQuantization.getZeroPoint(), input->getName());
                     std::vector<int32_t> inputZeroPoint(inZP.begin(), inZP.end());
                     std::transform(zeroPoint.begin(), zeroPoint.end(), inputZeroPoint.begin(), zeroPoint.begin(), std::minus<int32_t>());
                 }

                 auto m = S2;

                 if ((opIt->hasAttr("hasWeights") && opIt->get<bool>("hasWeights")) || isEltwiseMult)
                 {
                     auto weights = opIt->getInputTensor(1);
                     auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
                     scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
                     std::vector<float> S1(scale.begin(), scale.end());
                     //S1*S2
                     std::transform(m.begin(), m.end(), S1.begin(), m.begin(), std::multiplies<float>());
                 }
                 else if (isEltwiseAddSub) //Add Subtract
                 {
                     auto input2 = opIt->getInputTensor(1);
                     auto& input2Quantization = input2->get<mv::QuantizationParams>("quantParams");
                     auto input1Scale = inputQuantization.getScale();
                     auto input2Scale = input2Quantization.getScale();
                     
                    auto size = input1Scale.size();
                    std::vector <double> scaleDifference(size), absRelativeErrorScale(size), relativeErrorScale(size);
                    std::transform(input1Scale.begin(), input1Scale.end(), input2Scale.begin(), scaleDifference.begin(), std::minus<double>());

                    double (*fabs)(double) = &std::abs;
                    std::transform(scaleDifference.begin(), scaleDifference.end(), input1Scale.begin(), relativeErrorScale.begin(), std::divides<double>());
                    std::transform(relativeErrorScale.begin(),relativeErrorScale.end(), absRelativeErrorScale.begin(), fabs);
                    for (auto it = absRelativeErrorScale.begin(); it != absRelativeErrorScale.end(); it++)
                    {
                        if (*it > 0.01)
                            throw mv::RuntimeError(om, opIt->getName() + ": The relative difference in the input scales is > 1%. This is not supported for Eltwise operation."
                                               + std::to_string(input1Scale[0]) + " " + std::to_string(input2Scale[0]));
                    }
                 }

                 //Note: There are PPE Types SIGMOID, TAN, EXP, SQRT, RSQRT, FLEXARB that need their output
                 //quantized to 13-bits, then runtime uses a LUT to correspond to 8-bit
                 if (opIt->hasAttr("postOpTypes"))
                 {
                     auto ppeIterator = std::find(opIt->get<std::vector<std::string>>("postOpTypes").begin(),
                                               opIt->get<std::vector<std::string>>("postOpTypes").end(),
                                               "Sigmoid");
                     if (ppeIterator != opIt->get<std::vector<std::string>>("postOpTypes").end())
                     {
                        auto ppeQuantumBits = 5;
                        auto ppeQuantum = std::pow(2, ppeQuantumBits);
                        std::transform(m.begin(), m.end(), m.begin(), std::bind(std::multiplies<float>(),
                                                                                std::placeholders::_1, ppeQuantum));
                     }
                }
                 // Fuse ReLU into quantization (i.e. make ReLU == saturation), will be done using a separate pass

                 // m / S3
                 std::transform(m.begin(), m.end(), S3.begin(), m.begin(), std::divides<float>());

                 //TODO need to handle 16bits case - per Alessandro bias need to be converted to int32
                 auto bits = 15;
                 auto mSize = m.size();
                 int exponent;
                 double mantissa;

                 for (size_t i = 0; i < mSize; i++)
                 {
                     mantissa = std::frexp(m[i], &exponent);
                     shift[i] = bits - exponent;
                     mScaled[i] = (mantissa * pow(2, bits));
                 }
                 std::vector<int32_t> zeroPointScaled(m.size());
                 std::transform(zeroPoint.begin(), zeroPoint.end() , m.begin(), zeroPointScaled.begin(), std::divides<float>());

                 std::vector <unsigned> ser_shift = std::vector<unsigned>(shift.begin(), shift.end());
                 std::vector <unsigned> ser_scale = std::vector<unsigned>(mScaled.begin(), mScaled.end());
                 outputQuantization.quantize(ser_shift, ser_scale);
            }
        }

    }
}

template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

    // We enter in this case if and only if we specified multi channel scales and
    // the tensor has been aligned
    if (value.size() < size)
    {
        auto toReturn = mv::utils::generateSequence<T>(size, static_cast<T>(0) , 0);
        for(unsigned i = 0; i < value.size(); ++i)
            toReturn[i] = value[i];
        return toReturn;
    }

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}
