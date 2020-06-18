#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

void placeEltwiseDequantize(mv::OpModel om, mv::Data::OpListIterator task);
static void placementOfOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void addPermutesToChangeSoftmaxAxisFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void placeNeutralMaxPoolBefore(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::Element &, mv::Element &);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(PlaceNeutralMaxPoolBefore)
        .setFunc(placeNeutralMaxPoolBefore)
        .setDescription(
            "This pass handles a specific case in yoloV3, when an interpolate goes into a concat."
        );

        MV_REGISTER_PASS(PlacementOfOps)
        .setFunc(placementOfOps)
        .setDescription(
            "This pass handles the DPU's output Tensor Data Type."
        );

        MV_REGISTER_PASS(AddPermutesToChangeSoftmaxAxis)
        .setFunc(addPermutesToChangeSoftmaxAxisFcn)
        .setDescription(
            "UPA softmax layer does not support softmax oepration on the W axis. \
            The solution to this is to transpose the tensor before and after the softmax operation."
        );
    }
}

void placeNeutralMaxPoolBefore(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto resampleOps = om.getOps("Resample");

    for (auto& resample : resampleOps)
    {
        auto outputTensor = resample->getOutputTensor(0);
        auto nextOp = mv::findSinkLayers(dm, outputTensor)[0];
        if (nextOp->getOpType() == "Concat")
        {
            auto inputFlow = nextOp.leftmostInput();
            auto neutralMaxPool = om.maxPool(outputTensor, {1,1}, {1,1}, {0, 0, 0, 0}, false,
                mv::DType("UInt8"), outputTensor->get<mv::QuantizationParams>("quantParams"), nextOp->getName() + "MaxPool");
            auto maxPoolOp = om.getSourceOp(neutralMaxPool);
            maxPoolOp->set<unsigned>("opId", resample->get<unsigned>("opId"));
            while(inputFlow != om.flowEnd())
            {
                auto tensor = inputFlow->getTensor();
                if (tensor->getName() == outputTensor->getName())
                {
                    auto slot = inputFlow->get<size_t>("sinkInput");
                    om.undefineFlow(inputFlow);
                    nextOp->setInputTensor(neutralMaxPool, slot, false);
                    om.defineFlow(neutralMaxPool, nextOp, slot);
                }
            }
        }
    }
}

void addPermutesToChangeSoftmaxAxisFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    using namespace mv;
    OpModel om(model);
    auto softmaxOps = om.getOps("Softmax");

    for (auto& softmax : softmaxOps)
    {
        std::string axis = softmax->get<std::string>("axis");

        if (axis.compare(std::string("W")) == 0)
        {
            auto inputTensorSoftmax = softmax->getInputTensor(0);

            mv::QuantizationParams inputTensorSoftmaxQPs = {{},{},{},{}};
            if(inputTensorSoftmax->hasAttr("quantParams"))
                inputTensorSoftmaxQPs = inputTensorSoftmax->get<mv::QuantizationParams>("quantParams");

            // Permute acts on internal order WHCN
            mv::Order transposedOrder_swapWandH("HWCN");    // NCHW --> NCWH
            mv::Order transposedOrder_swapWHandC("CWHN");   // NCHW --> NHWC

            auto transposeBeforeSoftmax1 = om.permute(inputTensorSoftmax, transposedOrder_swapWandH, mv::DType("Default"), inputTensorSoftmaxQPs, softmax->getName() + "_permuteWandH");
            auto transposeBeforeSoftmax2 = om.permute(transposeBeforeSoftmax1, transposedOrder_swapWHandC, mv::DType("Default"), inputTensorSoftmaxQPs, softmax->getName() + "_permuteWHandC");
            auto transposeOpBeforeSoftmax1 = om.getSourceOp(transposeBeforeSoftmax1);
            auto transposeOpBeforeSoftmax2 = om.getSourceOp(transposeBeforeSoftmax2);
            auto outputMemoryLocationSoftmax = softmax->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            
            /*Remove old flow*/
            auto sourceFlow = softmax.leftmostInput();
            om.undefineFlow(sourceFlow);

            transposeOpBeforeSoftmax1->set<unsigned>("opId", softmax->get<unsigned>("opId"));
            transposeOpBeforeSoftmax2->set<unsigned>("opId", softmax->get<unsigned>("opId"));
            softmax->setInputTensor(transposeBeforeSoftmax2, 0, true);
            om.defineFlow(transposeBeforeSoftmax2, softmax, 0);
           
            transposeOpBeforeSoftmax1->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocationSoftmax);
            transposeOpBeforeSoftmax2->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocationSoftmax);
            softmax->set<std::string>("axis", "C");

            /*Permute after*/
            auto opAfterSoftmax = softmax.leftmostChild();
            auto inputTensorOpAfterSoftmax =  opAfterSoftmax->getInputTensor(0);
            mv::QuantizationParams inputTensorQuantizationParamsOpAfterSoftmax = {{},{},{},{}};
            if(inputTensorOpAfterSoftmax->hasAttr("quantParams"))
                inputTensorQuantizationParamsOpAfterSoftmax = inputTensorOpAfterSoftmax->get<mv::QuantizationParams>("quantParams");
            auto transposeAfterSoftmax1 = om.permute(inputTensorOpAfterSoftmax, transposedOrder_swapWHandC,  mv::DType("Default"), inputTensorQuantizationParamsOpAfterSoftmax, opAfterSoftmax->getName() + "_permuteWHandC1");
            auto transposeAfterSoftmax2 = om.permute(transposeAfterSoftmax1, transposedOrder_swapWHandC,  mv::DType("Default"), inputTensorQuantizationParamsOpAfterSoftmax, opAfterSoftmax->getName() + "_permuteWHandC2");
            auto transposeAfterSoftmax3 = om.permute(transposeAfterSoftmax2, transposedOrder_swapWandH,  mv::DType("Default"), inputTensorQuantizationParamsOpAfterSoftmax, opAfterSoftmax->getName() + "_permuteWandH");
            auto transposeOpAfterSoftmax1 = om.getSourceOp(transposeAfterSoftmax1);
            auto transposeOpAfterSoftmax2 = om.getSourceOp(transposeAfterSoftmax2);
            auto transposeOpAfterSoftmax3 = om.getSourceOp(transposeAfterSoftmax3);

            /*Remove old flow*/
            auto sourceFlow1 = opAfterSoftmax.leftmostInput();
            om.undefineFlow(sourceFlow1);
            transposeOpAfterSoftmax1->set<unsigned>("opId", opAfterSoftmax->get<unsigned>("opId"));
            transposeOpAfterSoftmax2->set<unsigned>("opId", opAfterSoftmax->get<unsigned>("opId"));
            transposeOpAfterSoftmax3->set<unsigned>("opId", opAfterSoftmax->get<unsigned>("opId"));
            opAfterSoftmax->setInputTensor(transposeAfterSoftmax3, 0, true);
            om.defineFlow(transposeAfterSoftmax3, opAfterSoftmax, 0);
            transposeAfterSoftmax1->set<mv::Tensor::MemoryLocation>("Location", inputTensorOpAfterSoftmax->get<mv::Tensor::MemoryLocation>("Location"));
            transposeAfterSoftmax2->set<mv::Tensor::MemoryLocation>("Location", inputTensorOpAfterSoftmax->get<mv::Tensor::MemoryLocation>("Location"));
            transposeAfterSoftmax3->set<mv::Tensor::MemoryLocation>("Location", inputTensorOpAfterSoftmax->get<mv::Tensor::MemoryLocation>("Location"));
        }
    }
}

void placeEltwiseDequantize(mv::OpModel om, mv::Data::OpListIterator task)
{
    auto neutralCopy = om.copy(task->getInputTensor(0), mv::DType("UInt8"),
                    task->getInputTensor(0)->get<mv::QuantizationParams>("quantParams"), task->getName() + "Neutral");
    auto neutralCopyOp = om.getSourceOp(neutralCopy);
    neutralCopyOp->set<unsigned>("opId", task->get<unsigned>("opId"));

    std::vector<mv::Data::TensorIterator> andInputs = {task->getInputTensor(0), neutralCopy};
    auto placeEltwiseDequantize = om.eltwise(andInputs, "And",
                                    mv::DType("Default"), {{0}, {1.0f}, {}, {}}, task->getName() + "AND_Conversion");
    auto placeEltwiseDequantizeOp = om.getSourceOp(placeEltwiseDequantize);

    placeEltwiseDequantizeOp->getInputTensor(0)->set<mv::DType>("dType", mv::DType("UInt8"));
    placeEltwiseDequantizeOp->getInputTensor(1)->set<mv::DType>("dType", mv::DType("UInt8"));
    placeEltwiseDequantizeOp->getOutputTensor(0)->set<mv::DType>("dType", mv::DType("Float16"));
    placeEltwiseDequantizeOp->set<bool>("mixedToFloat", true);

    placeEltwiseDequantizeOp->set<unsigned>("opId", task->get<unsigned>("opId"));
    task->setInputTensor(placeEltwiseDequantize, 0, false);
    om.defineFlow(placeEltwiseDequantize, task, 0);
}

void placementOfOps(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto convOps = om.getOps("Conv");
    auto convDepthwiseOps = om.getOps("DepthwiseConv");
    convOps.insert(convOps.end(),convDepthwiseOps.begin(),convDepthwiseOps.end());

    for (auto& opIt : convOps)
    {
        if (opIt->hasAttr("placeConversionToFloat"))
        {
            if (opIt->get<bool>("placeConversionToFloat"))
            {
                auto previousOpIt = om.getSourceOp(opIt->getInputTensor(0));
                bool conversionPlaced = false;
                std::vector<double> inputScale = opIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale();
                if (!previousOpIt->isUPA())
                {
                    conversionPlaced = true;
                    placeEltwiseDequantize(om, opIt);
                }
                //NOTE: For now take for granted that the next guy is a convolution
                opIt->set<bool>("floatPrecision", true);
                opIt->set<mv::DType>("dType", mv::DType("Float16"));
                //NOTE: Do not care of the data type of input but it will be float so all
                //inputs and outputs need to be converted to float, populated need de-quantize!!!
                for(std::size_t i = 0; i < opIt->inputSlots(); ++i)
                    opIt->getInputTensor(i)->set<mv::DType>("dType", mv::DType("Float16"));
                opIt->getOutputTensor(0)->set<mv::DType>("dType", mv::DType("Float16"));
//                opIt->set<mv::DType>("dType", mv::DType("Float16"));
                bool hasBias = opIt->hasAttr("bias");

                // If this conv was a tiled conv, pass the conversion to the adds as well
                if(opIt->hasAttr("partitionedKernelToAdd"))
                {
                    if(opIt->get<bool>("partitionedKernelToAdd"))
                    {
                        auto partitionAdd = opIt.leftmostOutput().sink();
                        partitionAdd->getOutputTensor(0)->set<mv::DType>("dType", mv::DType("Float16"));
                    }
                }

                if (opIt->hasWeights())
                {
                    double real_weight;
                    double real_weight_fp16;
                    mv::Data::TensorIterator weightsTensor =  opIt->getInputTensor(1);
                    std::vector<int64_t> weightsData(weightsTensor->getData().size());
                    std::vector<double> scale = weightsTensor->get<mv::QuantizationParams>("quantParams").getScale();
                    std::vector<int64_t> zp = weightsTensor->get<mv::QuantizationParams>("quantParams").getZeroPoint();
                    auto kernelShape = opIt->getInputTensor(1)->getShape();
                    scale = extendToK(kernelShape[mv::KERNEL_OUTPUT_CHANNELS], scale, weightsTensor->getName());
                    zp = extendToK(kernelShape[mv::KERNEL_OUTPUT_CHANNELS], zp, weightsTensor->getName());

                    for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
                    {
                        for (size_t c = 0; c < kernelShape[mv::KERNEL_INPUT_CHANNELS]; c++)
                        {
                            for (size_t h = 0; h < kernelShape[mv::KERNEL_HEIGHT]; h++)
                            {
                                for (size_t w = 0; w < kernelShape[mv::KERNEL_WIDTH]; w++)
                                {
                                    auto currWeight = (int64_t)weightsTensor->at({w,h,c,k});
                                    real_weight = ((int64_t)currWeight - zp[k]) * scale[k];
                                    real_weight_fp16 = mv::fp32_to_fp16(real_weight);
                                    const size_t idx = (k * kernelShape[mv::KERNEL_INPUT_CHANNELS] * kernelShape[mv::KERNEL_WIDTH] * kernelShape[mv::KERNEL_HEIGHT]) +
                                                       (c * kernelShape[mv::KERNEL_WIDTH] * kernelShape[mv::KERNEL_HEIGHT]) +
                                                       (h * kernelShape[mv::KERNEL_WIDTH]) +
                                                        w;
                                    weightsData[idx] = real_weight_fp16;
                                }
                            }
                        }
                    }
                    auto weights = om.constantInt(weightsData,
                                        {kernelShape[mv::KERNEL_WIDTH], kernelShape[mv::KERNEL_HEIGHT],
                                        kernelShape[mv::KERNEL_INPUT_CHANNELS], kernelShape[mv::KERNEL_OUTPUT_CHANNELS]},
                                        mv::DType("Float16"),
                                        weightsTensor->getOrder(),
                                        {{0},{1},{},{}}, opIt->getName() + "FP16_weights");
                    if (hasBias)
                    {
                        mv::Data::TensorIterator bias =  dm.getTensor(opIt->get<std::string>("bias"));
                        auto outputShape = opIt->getOutputTensor(0)->getShape();
                        std::vector<int64_t> biasData;
                        double biasOldScale, real_bias;
                        int64_t real_bias_fp16;
                        std::vector<double> weightsScale = opIt->getInputTensor(1)->get<mv::QuantizationParams>("quantParams").getScale();
                        weightsScale = extendToK(outputShape[mv::IO_CHANNEL_DIMENSION], weightsScale, bias->getName());
                        for (size_t k = 0; k < outputShape[mv::IO_CHANNEL_DIMENSION]; k++)
                        {
                            biasOldScale = weightsScale[k] * inputScale[0];
                            real_bias = ((int64_t) bias->at(k)) * biasOldScale;
                            real_bias_fp16 = mv::fp32_to_fp16(real_bias);
                            biasData.push_back(real_bias_fp16);
                        }
                        mv::Data::TensorIterator floatBias;
                        std::string floatBiasName = mv::createBiasName(opIt->getName() + "FP16_bias");
                        floatBias = dm.defineTensor(mv::Tensor(floatBiasName, bias->getShape(),
                                                     mv::DType("Float16"), bias->getOrder(), biasData, {{0},{1},{},{}}));
                        om.eraseAttr(opIt, "bias");
                        om.addAttr(opIt, "bias", floatBiasName);
                        bias->set<mv::DType>("dType", mv::DType("Float16"));
                    }
                    if (conversionPlaced)
                    {
                        for (auto sourceFlow = opIt.leftmostInput(); sourceFlow != om.flowEnd(); ++sourceFlow)
                        {
                            if (sourceFlow.source()->getName() == previousOpIt->getName())
                                om.undefineFlow(sourceFlow);
                        }
                    }
                    om.removeOp(om.getSourceOp(opIt->getInputTensor(1)));
                    opIt->setInputTensor(weights, 1, false);
                    om.defineFlow(weights, opIt, 1);
                    om.getSourceOp(opIt->getInputTensor(1))->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                }
            }
        }
    }
}
