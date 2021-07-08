#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

void placeHwDequantize(mv::OpModel & om, mv::Data::OpListIterator task);
static void placementOfOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void placeNeutralMaxPoolBefore(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::Element &, mv::Element &);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(PlaceNeutralMaxPoolBefore)
        .setFunc(placeNeutralMaxPoolBefore)
        .setDescription(
            "This pass handles a specific case in yoloV3/Unet/Deblur, when an UPA Op goes into a concat."
        );

        MV_REGISTER_PASS(PlacementOfOps)
        .setFunc(placementOfOps)
        .setDescription(
            "This pass handles the DPU's output Tensor Data Type."
        );
    }
}

void placeNeutralMaxPoolBefore(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    auto concats = om.getOps("Concat");
    const auto globalParams = model.getGlobalConfigParams();
    const bool allowFloatOutput =
        globalParams->hasAttr("FloatOutput") && globalParams->get<bool>("FloatOutput");

    for (auto& concatOp : concats)
    {
        const auto numInputs = concatOp->getInputTensor().size();
        unsigned short numUPAOps = 0;

        for (size_t i = 0; i < numInputs; i++)
        {
            const auto sourceOp = om.getSourceOp(concatOp->getInputTensor(i));
            if (sourceOp->isUPA() || sourceOp->getOpType() == "UPATask")
                numUPAOps++;
        }
        if (numUPAOps == 0 || numUPAOps == numInputs) //no mixed upa + dpu
            continue;

        for (size_t i = 0; i < numInputs; i++)
        {
            const auto sourceOp = om.getSourceOp(concatOp->getInputTensor(i));
            if (sourceOp->isUPA() || sourceOp->getOpType() == "UPATask")
            {
                auto inputFlow = concatOp.leftmostInput();
                const auto outputTensor = sourceOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
                auto neutralMaxPool = om.maxPool(concatOp->getName() + "MaxPool" + std::to_string(i), outputTensor, {1,1}, {1,1}, {0, 0, 0, 0}, false);
                neutralMaxPool->setQuantParams(outputTensor->getQuantParams());
                auto maxPoolOp = om.getSourceOp(neutralMaxPool);
                maxPoolOp->set<unsigned>("opId", sourceOp->get<unsigned>("opId"));

                if (neutralMaxPool->getDType() == mv::DType("Float16") && allowFloatOutput) {
                    maxPoolOp->set<bool>("floatPrecision", true);
                }

                while(inputFlow != om.flowEnd())
                {
                    auto tensor = inputFlow->getTensor();
                    if (tensor->getName() == outputTensor->getName())
                    {
                        const auto slot = inputFlow->get<size_t>("sinkInput");
                        om.undefineFlow(inputFlow);
                        concatOp->setInputTensor(neutralMaxPool, slot, false);
                        om.defineFlow(neutralMaxPool, concatOp, slot);
                        break;
                    }
                    ++inputFlow;
                }
            }
        }
    }

}

bool isOpSoftware(const mv::Data::OpListIterator& opIt) {
    return opIt->isUPA() ||
           (opIt->hasAttr("softwareExecuted") && opIt->get<bool>("softwareExecuted")) ||
           (opIt->hasAttr("floatPrecision") && opIt->get<bool>("floatPrecision")) ||
           (opIt->hasAttr("placeConversionToFloat") && opIt->get<bool>("placeConversionToFloat"));
}

std::vector<mv::Data::OpListIterator> findNonImplicitConsumerOps(mv::DataModel& dm, const mv::Data::TensorIterator& tensor)
{
    std::vector<mv::Data::OpListIterator> consumerOps;
    const auto sinkLayers = mv::findSinkLayers(dm, tensor);
    for (auto& sink : sinkLayers)
    {
        if (sink->isImplicit() || sink->getOpType() == "Concat")
            for (auto& outputTensor : sink->getOutputTensor())
            {
                auto consumers = findNonImplicitConsumerOps(dm, outputTensor);
                consumerOps.insert(consumerOps.end(), consumers.begin(), consumers.end());
            }
        else
            consumerOps.push_back(sink);
    }
    return consumerOps;
}

void placeInputHwDequantize(mv::OpModel& om, mv::DataModel& dm, mv::Data::OpListIterator& opIt)
{
    auto allConsumersNeedFloat = [&dm](const mv::Data::TensorIterator& tensor) {
        const auto consumerOps = findNonImplicitConsumerOps(dm, tensor);
        for (auto& consumerOp : consumerOps) {
            if (!isOpSoftware(consumerOp))
                return false;
        }
        return true;
    };

    for (size_t i = 0; i < (opIt->getOpType() == "Eltwise" ? 2 : 1); ++i)
    {
        const auto inputTensor = opIt->getInputTensor(i);
        if (inputTensor->isFloatingPointType())
            continue;

        auto parentOp = om.getSourceOp(inputTensor);
        if (isOpSoftware(parentOp) && allConsumersNeedFloat(inputTensor))
            continue;

        if (parentOp->hasAttr("placeConversionToFloat") && parentOp->get<bool>("placeConversionToFloat"))
            continue;

        auto inputFlow = opIt.leftmostInput();
        while (inputFlow != om.flowEnd())
        {
            if (inputFlow->getTensor()->getName() == inputTensor->getName())
                break;
            ++inputFlow;
        }

        // First check if parent op on other output branches has already HwConvert op in place
        // which can be reused. In such case just attach to output of this HwConvert
        for(auto childOp = parentOp.leftmostChild(); childOp != om.opEnd(); ++childOp)
        {
            auto opType = childOp->getOpType();
            auto childOpOutput = childOp->getOutputTensor(0);
            if (opType == "HwConvert" && childOpOutput->getDType() == mv::DType("Float16"))
            {
                // Connect to this op output
                om.undefineFlow(inputFlow);
                opIt->setInputTensor(childOpOutput, i, false);
                om.defineFlow(childOpOutput, opIt, i);
                return;
            }
        }

        // Insert HwConvert operation to perform U8->FP16 conversion on DPU
        auto placeHwConvert = om.hwConvert(opIt->getName() + "_dequantize" + std::to_string(i), inputTensor, mv::DType("Float16"));

        auto placeHwConvertOp = om.getSourceOp(placeHwConvert);
        placeHwConvertOp->set<bool>("mixedToFloat", true);
        placeHwConvertOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));

        om.undefineFlow(inputFlow);
        opIt->setInputTensor(placeHwConvert, i, false);
        om.defineFlow(placeHwConvert, opIt, i);
    }
}

void placeOutputQuantize(mv::OpModel& om, mv::DataModel& dm, const mv::Data::OpListIterator& opIt) {
    const auto outputTensor = opIt->getOutputTensor(0);
    // originalDataType
    if (outputTensor->isFloatingPointType() && (!outputTensor->hasAttr("originalDataType")))
        return;

    if (outputTensor->hasAttr("originalDataType"))
    {
        mv::DType originalType = outputTensor->get("originalDataType");
        if (originalType.isDoubleType() || originalType == mv::DType("Float16") || originalType == mv::DType("BFloat16"))
            return;
    }

    const auto consumerOps = mv::findSinkLayers(dm, outputTensor);
    if (std::all_of(consumerOps.begin(), consumerOps.end(), isOpSoftware))
        return;

    auto quantize = om.quantize(opIt->getName() + "_Quantize", {outputTensor});
    quantize->setQuantParams(outputTensor->getQuantParams());

    auto quantizeOp = om.getSourceOp(quantize);
    quantizeOp->getInputTensor(0)->setDType(mv::DType("Float16"));
    quantizeOp->getInputTensor(0)->setQuantParams(mv::QuantizationParams::initial());
    quantizeOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));

    for (auto consumerOp : consumerOps)
    {
        if (consumerOp->hasAttr("placeConversionToFloat") && consumerOp->get("placeConversionToFloat"))
            continue;

        std::size_t i = 0;
        for (; i < consumerOp->inputSlots(); ++i)
        {
            if (consumerOp->getInputTensor(i)->getName() == outputTensor->getName())
                break;
        }
        auto inputFlow = consumerOp.leftmostInput();
        while (inputFlow != om.flowEnd())
        {
            if (inputFlow->getTensor()->getName() == outputTensor->getName())
                break;
            ++inputFlow;
        }
        om.undefineFlow(inputFlow);
        consumerOp->setInputTensor(quantize, i, false);
        om.defineFlow(quantize, consumerOp, i);
    }
}

void placementOfOps(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    // There is current list of the layers which might be converted to Float16 or BFloat16
    // if the option 'placeConversionToFloat' was set previously in them.
    auto opTypes = om.getOpsOfTypes({"Conv", "DepthwiseConv", "MaxPool", "Eltwise"});

    for (auto& opType: opTypes)
    {
        for (auto& opIt : opType.second)
        {
            if (!opIt->hasAttr("placeConversionToFloat") || !opIt->get<bool>("placeConversionToFloat"))
                continue;

            auto targetDType = mv::DType("Float16");
            if (opIt->getInputTensor(0)->getDType() == mv::DType("BFloat16") || opIt->getOutputTensor(0)->getDType() == mv::DType("BFloat16"))
                targetDType = mv::DType("BFloat16");

            placeInputHwDequantize(om, dm, opIt);
            placeOutputQuantize(om, dm, opIt);

            opIt->set<bool>("floatPrecision", true);
            for (size_t i = 0; i < (opIt->getOpType() == "Eltwise" ? 2 : 1); ++i) {
                auto parentOp = om.getSourceOp(opIt->getInputTensor(i));
                if (parentOp->hasAttr("placeConversionToFloat") && parentOp->get<bool>("placeConversionToFloat"))
                {
                    // Handle the case where the ops connect as:
                    // Eltwise1_float->ops_u8; Eltwise1_float->Eltwise2_FP16; ops_u8->Eltwise2_float.
                    // Save the originalDataType for the output tensor of Eltwise1_float
                    // to make sure a quantization will be inserted between Eltwise1_float->ops_u8
                    if (!opIt->getInputTensor(i)->hasAttr("originalDataType"))
                        opIt->getInputTensor(i)->set<mv::DType>("originalDataType", opIt->getInputTensor(i)->getDType());
                }
                else
                {
                    opIt->getInputTensor(i)->setDType(targetDType);
                    opIt->getInputTensor(i)->setQuantParams(mv::QuantizationParams::initial());
                }
            }
            opIt->getOutputTensor(0)->setDType(targetDType);
            opIt->getOutputTensor(0)->setQuantParams(mv::QuantizationParams::initial());

            // If this conv was a tiled conv, pass the conversion to the adds as well
            if (opIt->hasAttr("partitionedKernelToAdd"))
            {
                if (opIt->get<bool>("partitionedKernelToAdd"))
                {
                    auto partitionAdd = opIt.leftmostOutput().sink();
                    partitionAdd->getOutputTensor(0)->setDType(targetDType);
                }
            }

            if (opIt->hasWeights())
            {
                const auto weightsTensor = opIt->getInputTensor(1);
                if (weightsTensor->isFloatingPointType())
                    continue;
                const auto dequantFP16Weights = dequantizeWeightsToFP16(weightsTensor, opIt, om);

                if (opIt->hasAttr("bias"))
                {
                    const auto outputShape = opIt->getOutputTensor(0)->getShape();
                    auto bias = dm.getTensor(opIt->get<std::string>("bias"));
                    // hack
                    std::vector<double> outputScale = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale();
                    outputScale = extendToK(outputShape[mv::IO_CHANNEL_DIMENSION], outputScale, opIt->getOutputTensor(0)->getName());

                    auto biasScale = bias->getQuantParams().getScale();
                    std::vector<int64_t> biasData;
                    for (size_t k = 0; k < outputShape[mv::IO_CHANNEL_DIMENSION]; k++)
                    {
                        double scale = biasScale[k];
                        if (opIt->hasAttr("biasOverflow") && opIt->get<bool>("biasOverflow")) {
                            scale *= outputScale[k];
                        }
                        const double realBias = ((int64_t) bias->at(k)) * scale;
                        const int64_t realBiasFp16 = mv::fp32_to_fp16(realBias);
                        biasData.push_back(realBiasFp16);
                    }
                    const auto floatBiasName = mv::createBiasName(opIt->getName() + "FP16_bias");
                    const auto floatBias = dm.defineTensor(mv::Tensor(floatBiasName, bias->getShape(),
                                                           mv::DType("Float16"), bias->getOrder(), biasData, {{0},{1},{},{}}));
                    om.eraseAttr(opIt, "bias");
                    om.addAttr(opIt, "bias", floatBiasName);
                    bias->setDType(mv::DType("Float16"));
                }
                om.removeOp(om.getSourceOp(weightsTensor));
                opIt->setInputTensor(dequantFP16Weights, 1, false);
                om.defineFlow(dequantFP16Weights, opIt, 1);
                om.getSourceOp(opIt->getInputTensor(1))->set<unsigned>("opId", opIt->get<unsigned>("opId"));
            }
        }
    }
}
