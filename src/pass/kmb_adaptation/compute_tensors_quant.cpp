#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

static void computeTensorsQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void postTrainingQuantize(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignConcatScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void placeReQuantizeDepthwiseBefore(mv::OpModel om, mv::Data::OpListIterator concat, mv::Data::TensorIterator inputTensor, std::size_t index, double &weightScale, double &alignedScale, int64_t &alignedZeroPoint);
//static void compensateDepthWiseAfter(mv::OpModel om, mv::Data::OpListIterator nextOp, mv::Data::OpListIterator concat);
//static std::vector<mv::Data::OpListIterator> findNextConcat(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor);

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
    std::vector<double> minInputFloats, maxInputFloats;

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
    std::vector<mv::Data::OpListIterator> childConcats;
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

// A1) Normal U8 quant output
// Compute_stage -> I32 -> Requant_stage * ((in_sc * wt_sc)/out_sc)
//      -> I8 -> + ZP -> U8
//
// HW does not support scale dequantization at operation input,
// because that would require to store dequant input data in float format
// while the HW compute pipeline is integer. So we apply input and weights
// scale alongside output scale in a requantization stage which will take
// integer values in I32 range and output int values in the requested output
// range, I8 , I13 ...
// Lets disect the requant stage: between the *(in_sc*wt_sc) and / out_sc
// operations, out datatype would be float, and / out_sc will quantize it
// to whatever range we wish.
// Requant_stage: I32 -> * (in_sc * wt_sc) -> FP -> / out_sc -> I8
// We cant store float values the pipeline so we always apply
// in_wt_sc and out_sc at the same time.
//
// B1) Normal F16 output
// Compute_stage -> S16.16 -> Requant_stage * ((in_sc * wt_sc)/out_sc)
//      -> S16.16 -> Pack_stage -> FP16
//
// We support FP16 multiplications but all the results are still stored
// in an integer fixed point accumualator of form S16.16
// HW is flexible about using setting in_sc, wt_sc and out_sc in
// such a way to support also mixed precision cases U8->FP16 or FP16->U8
//
// A2) U8 quant output + PWL table (I13 input and output)
// Compute_stage -> I32 -> Requant_stage * ((in_sc * wt_sc)/out_sc) -> CLAMP
//      -> I13 -> -> PWL -> I13 -> PostShift_stage << >> -> I8 -> + ZP -> U8
//
// The PWL table requires a fixed input and output quantization
// Same applies for both fixed and custom HW tables.
// Custom tables can be trained to support a optimal quantization range.
// So let's take the HW fixed Sigmoid table for example, we know it works
// best with [-4.0, 4.0] input range and [0.0, 1.0] output range.
// Also because PWL input is placed after the requantization stage we must ensure correct
// quantization range at requantization. For this we drop completely the fused
// (conv|pool|eltwise + sigmoid) quantization from the model and enforce the
// fixed one that PWL needs.
// For the U8 Sigmoid case we need to map [-4.0, 4.0] to the input of PWL which is
// [-4096, 4095] I13. To achieve this we set output scale to 1/1024 and clamp to
// [-4096, 4095].
// At the SIGMOID table output we have an I13 result of [0; 4095] which maps to [0.0, 1.0]
// float. Our only way to translate I13 to either I8, U8 or S16.16 is via the final post_shift
// stage which is bidirectional << >>. So we'll do [0; 4095] >> 4 -> [0, 255]
// The same forced quantization idea needs to be applied for the input tensor of the
// next comsuming operations. In this case we'll need to force a quantization of
// input float range [0.0, 1.0] with zp 0 and scale 1 / 255
//
// B2) FP16 quant output + PWL table (I13 input and output)
// Compute_stage -> S16.16 -> Requant_stage * ((in_sc * wt_sc)/out_sc) -> CLAMP
//      -> I13 (S3.10) -> PWL -> I13 (S1.12) -> PostShift_stage << >> -> S16.16 -> + ZP -> U8
//
// We can also support PWL table for FP16 output cases by taking into account the required
// fixed quantizations for table input and output.
// For example Sigmoid table input needs [-4.0, 4.0] which is achieved by clamping S16.16
// to (S3.16) and shifting >> 6 to match the I13 bits requirements for a final S3.10
// fixed point number.
// Additionally we take the I13 output of range [0.0, 1.0] and consider it a S1.12 fixed
// point result. Lastly we need to shift back to S16.16 which is done via post shift << 4.

template <typename K, typename V, typename H = std::hash<K>>
using map = std::unordered_map<K, V, H>;
using pwlFixedQuantEntry = std::vector<mv::QuantizationParams>;

void loadPWLQuantParams(const mv::pass::PassEntry& pass, mv::Data::OpListIterator& op) {

    // The preprogrammed hw PWL functions have fixed quantization requirements
    // on input and output of the activation function
    const map<std::string, map<std::string, pwlFixedQuantEntry>> pwlFixedQuantization =
        {
        {
            "Sigmoid",
            {
            {
                "UInt8",
                {
                mv::QuantizationParams({0}, {1.0 / 1015.6875}, {-4.0}, {4.0}, {0}, {1}, {4}),
                // fine tuning showed that 1/1015.6875 had the best sigmoid avg precision
                // in quantization case, better than the theoretical 1 / 1024
                mv::QuantizationParams({3}, {1.0 / 249}, {0}, {1.0})
                // to account better for sigmoid 0.0 and 1.0 results, we tweak zp and scale
                // to 3 and 1.0/249
                }
            },
            {
                "Float16",
                {
                mv::QuantizationParams({0}, {1.0 / 64}, {-4.0}, {4.0}, {0}, {1}, {-4}),
                mv::QuantizationParams({0}, {1.0}, {0}, {1.0})
                }
            }
            }
        },
        {
            "Tanh",
            {
            {
                "UInt8",
                {
                mv::QuantizationParams({0}, {1.0 / 1024}, {-4.0}, {4.0}, {0}, {1}, {5}),
                mv::QuantizationParams({128}, {1.0 / 127}, {-1.0}, {1.0})
                }
            },
            {
                "Float16",
                {
                mv::QuantizationParams({0}, {1.0 / 64}, {-4.0}, {4.0}, {0}, {1}, {-4}),
                mv::QuantizationParams({0}, {1.0}, {-1.0}, {1.0})
                }
            }
            }
        }
        };

    auto actDType = op->getInputTensor(0)->getDType().toString();

    //Note: There are PPE Types SIGMOID, TANH, EXP, SQRT, RSQRT, FLEXARB that need their output
    //quantized to 13-bits for LUT entry, then runtime converts the LUT output to the expected dtype
    if (op->hasAttr("postOpTypes"))
    {
        std::string resolvedPWL;
        for (auto postOp : op->get<std::vector<std::string>>("postOpTypes"))
        {
            auto fixedQuantIt = pwlFixedQuantization.find(postOp);
            if (fixedQuantIt != pwlFixedQuantization.cend())
            {

                if (!resolvedPWL.empty())
                {
                    pass.log(mv::Logger::MessageType::Warning,
                        "Multiple PWL functions associated per DPU task: " +
                        fixedQuantIt->first + " is ignored since " + resolvedPWL + " is already set");
                    continue;
                }

                auto typedFixedQuantIt = fixedQuantIt->second.find(actDType);

                if (typedFixedQuantIt == fixedQuantIt->second.cend()) {
                    pass.log(mv::Logger::MessageType::Error,
                        "No fixed quantization scheme associated for " +
                        fixedQuantIt->first + " with datatype " + actDType);
                    continue;
                }

                auto pwlInputQuant = typedFixedQuantIt->second[0];
                auto pwlOutputQuant = typedFixedQuantIt->second[1];

                // Use the pwl input scale to futher compute mult and shift
                op->set<mv::QuantizationParams>("pwlQuantParams", pwlInputQuant);

                for(auto output : op->getOutputTensor()){
                    if (!output->isQuantized())
                        continue;
                    output->set<mv::QuantizationParams>("quantParams", pwlOutputQuant);
                }

                resolvedPWL = postOp;
            }
        }
    }
}

void computeQuantMultShift(
    std::vector<float>& scale,
    std::vector<unsigned>& shift,
    std::vector<unsigned>& mult)
{
    //TODO need to handle 16bits case - per Alessandro bias need to be converted to int32
    auto bits = 15;
    auto scaleSize = scale.size();
    int exponent;
    double mantissa;

    {
        for (size_t i = 0; i < scaleSize; i++)
        {
            mantissa = std::frexp(scale[i], &exponent);
            shift[i] = bits - exponent;
            mult[i] = (mantissa * pow(2, bits));
        }
    }
}

void updatePWLQuantParams(mv::Data::OpListIterator& op,
    std::vector<float>& inputScale) {

    if(!op->hasAttr("pwlQuantParams"))
        return;

    auto pwlQuant = op->get<mv::QuantizationParams>("pwlQuantParams");

    auto reQuantScale = std::vector<float>(inputScale.begin(), inputScale.end());
    auto quantSize = reQuantScale.size();

    pwlQuant.setScale(
        extendToK(quantSize, pwlQuant.getScale(), op->getName()));
    pwlQuant.setZeroPoint(
        extendToK(quantSize, pwlQuant.getZeroPoint(), op->getName()));

    // (input_scale * weight_scale) / output_scale
    std::transform(
        reQuantScale.begin(),
        reQuantScale.end(),
        pwlQuant.getScale().begin(),
        reQuantScale.begin(),
        std::divides<float>());

    std::vector<unsigned> reQuantShift(quantSize, 0);
    std::vector<unsigned> reQuantMult(quantSize, 1);

    computeQuantMultShift(reQuantScale, reQuantShift, reQuantMult);

    op->get<mv::QuantizationParams>("pwlQuantParams").quantize(reQuantShift, reQuantMult);
}

void computeTensorsQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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

        loadPWLQuantParams(pass, opIt);

        if(isEltwise)
        {
            auto eltwiseType = opIt->get<std::string>("eltwiseType");
            if(eltwiseType == "Add" || eltwiseType == "Subtract" || eltwiseType == "And")
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

            std::vector<unsigned> shift(outputChannels, 0);
            std::vector<unsigned> mult(outputChannels, 0);

            if (output->isQuantized() && input->isQuantized())
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

                std::vector <float> S3(outputChannels, 1);
                std::vector <float> floatScale(outputChannels, std::pow(2, -16));
                std::vector <int64_t> zeroPoint(outputChannels, 0);
                bool outputOfAccWithBias = true;
                mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                if (!(output->hasAttr("dType") && output->get<mv::DType>("dType") == mv::DType("Int32")))
                {
                    //NOTE: Here I compute all the quantization parameters like they should be
                    //in order to dequantize the output of the DPU TASK, as Int32
                    //32 bit is not a correct statement, practically it is Int33 as
                    //the output of the accumulator+bias is an int33 number, but in
                    //graphfile and everywhere will be noted as Int32 for better exposing to the user
                    outputOfAccWithBias = false;
                    scale = extendToK(outputChannels, outputQuantization.getScale(), output->getName());
                    S3 = {scale.begin(), scale.end()};

                    auto zeroPointU =  extendToK(outputChannels, outputQuantization.getZeroPoint(), output->getName());
                    zeroPoint = {zeroPointU.begin(), zeroPointU.end()};
                }
                if (opIt->hasAttr("mixedToFloat") && opIt->get<bool>("mixedToFloat"))
                    S3 = {floatScale.begin(), floatScale.end()};

                bool isPooling = taskOp == "MaxPool";
                //Workaround for HW bug #227
                if (isPooling)
                {
                    auto inZP = extendToK(outputChannels, inputQuantization.getZeroPoint(), input->getName());
                    std::vector<int64_t> inputZeroPoint(inZP.begin(), inZP.end());
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
                    if (output->hasAttr("dType") && output->get<mv::DType>("dType") == mv::DType("Int32"))
                    {
                        std::vector<double> output_scale;
                        output_scale = inputQuantization.getScale();
                        std::transform(output_scale.begin(), output_scale.end(),
                                    weightsQuantization.getScale().begin(), output_scale.begin(), std::multiplies<double>());
                        outputQuantization.setScale(output_scale);
                    }

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

                updatePWLQuantParams(opIt, m);

                // m / S3
                std::transform(m.begin(), m.end(), S3.begin(), m.begin(), std::divides<float>());

                if (outputOfAccWithBias)
                {
                    for (size_t i = 0; i < m.size(); i++)
                    {
                        shift[i] = 0;
                        mult[i] = 1;
                    }
                }
                else
                    computeQuantMultShift(m, shift, mult);

                outputQuantization.quantize(shift, mult);
                 if (opIt->hasAttr("postOpTypes"))
                 {
                     signed postShift = 0;
                     auto ppeFlexARBdIterator = std::find(opIt->get<std::vector<std::string>>("postOpTypes").begin(),
                                               opIt->get<std::vector<std::string>>("postOpTypes").end(),
                                               "FLEXARB");
                     if (ppeFlexARBdIterator != opIt->get<std::vector<std::string>>("postOpTypes").end())
                        postShift = 4;
                     mv::QuantizationParams postQuantization = {{outputQuantization.getZeroPoint()},{outputQuantization.getScale()},
                                                                {outputQuantization.getMin()},{outputQuantization.getMax()},
                                                                ser_shift, ser_scale, postShift};
                    output->set<mv::QuantizationParams>("quantParams", postQuantization);
                 }

            }
        }

    }
}

