#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

void placeEltwiseDequantize(mv::OpModel om, mv::Data::OpListIterator task);
static void placementOfOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void changeSoftmaxAxis(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(PlacementOfOps)
        .setFunc(placementOfOps)
        .setDescription(
            "This pass handles the DPU's output Tensor Data Type."
        );

        MV_REGISTER_PASS(ChangeSoftmaxAxis)
        .setFunc(changeSoftmaxAxis)
        .setDescription(
            "This pass reshape tensors for softmax if the softmax axis is W, which is not supported"
        );
    }
}

void changeSoftmaxAxis(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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
            auto parentOpOfSoftmax = om.getSourceOp(inputTensorSoftmax);

            mv::QuantizationParams inputTensorSoftmaxQPs = inputTensorSoftmax->get<mv::QuantizationParams>("quantParams");
            auto inputTensorTypeSoftmax = inputTensorSoftmax->get<mv::DType>("dType");
            auto outputTensorTypeSoftmax = softmax->getOutputTensor(0)->get<mv::DType>("dType");

            mv::Shape outputShape = {softmax->getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION], softmax->getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION], softmax->getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION], softmax->getInputTensor()[0]->getShape()[mv::IO_BATCH_DIMENSION]};
            
            auto reshapeBeforeSoftmax = om.reshape(inputTensorSoftmax, outputShape, inputTensorTypeSoftmax, inputTensorSoftmaxQPs,  softmax->getName() + "_reshape");
            auto reshapeOpBeforeSoftmax = om.getSourceOp(reshapeBeforeSoftmax);
            auto outputMemoryLocationSoftmax = softmax->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            
            /*Remove old flow*/
            auto sourceFlow = softmax.leftmostInput();
            om.undefineFlow(sourceFlow);

            reshapeOpBeforeSoftmax->set<unsigned>("opId", softmax->get<unsigned>("opId"));
            softmax->setInputTensor(reshapeBeforeSoftmax, 0, true);
            om.defineFlow(reshapeBeforeSoftmax, softmax, 0);
           
            reshapeBeforeSoftmax->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocationSoftmax);
            softmax->set<std::string>("axis", "H");

            /*Reshape after*/
            auto opAfterSoftmax = softmax.leftmostChild();
            auto inputTensorOpAfterSoftmax =  opAfterSoftmax->getInputTensor(0);
            mv::QuantizationParams inputTensorQuantizationParamsOpAfterSoftmax = inputTensorOpAfterSoftmax->get<mv::QuantizationParams>("quantParams");
            auto inputTensorType1 = inputTensorOpAfterSoftmax->get<mv::DType>("dType");
            auto outputTensorType1 = softmax->getOutputTensor(0)->get<mv::DType>("dType");
            mv::Shape outputShape1 = {softmax->getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION], softmax->getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION], softmax->getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION], softmax->getInputTensor()[0]->getShape()[mv::IO_BATCH_DIMENSION]};
            auto reshapeAfterSoftmax = om.reshape(inputTensorOpAfterSoftmax, outputShape1, outputTensorType1, inputTensorQuantizationParamsOpAfterSoftmax,  opAfterSoftmax->getName() + "_reshape");
            auto reshapeOpAfterSoftmax = om.getSourceOp(reshapeAfterSoftmax);
            
            /*Remove old flow*/
            auto sourceFlow1 = opAfterSoftmax.leftmostInput();
            om.undefineFlow(sourceFlow1);
            reshapeOpAfterSoftmax->set<unsigned>("opId", opAfterSoftmax->get<unsigned>("opId"));
            opAfterSoftmax->setInputTensor(reshapeAfterSoftmax, 0, true);
            om.defineFlow(reshapeAfterSoftmax, opAfterSoftmax, 0);
            reshapeAfterSoftmax->set<mv::Tensor::MemoryLocation>("Location", inputTensorOpAfterSoftmax->get<mv::Tensor::MemoryLocation>("Location"));
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
                std::vector<double> inputScale = opIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale();
                placeEltwiseDequantize(om, opIt);
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
                    for (auto sourceFlow = opIt.leftmostInput(); sourceFlow != om.flowEnd(); ++sourceFlow)
                    {
                        if (sourceFlow.source()->getName() == previousOpIt->getName())
                            om.undefineFlow(sourceFlow);
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
