#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>

static const uint8_t ADD_INPUT_FLOWS = 2;
static const double max_inf = std::numeric_limits<double>::infinity();
static const double min_inf = -std::numeric_limits<double>::infinity();

static mv::QuantizationParams adjustFQtoHalfSum(const mv::QuantizationParams &quantParams, bool toNegative)
{
    // recalc quant params to only positive part, extend max (we has unused negative part) to keep as many values as possible
    // its important because we divide convolution in 2 halfsum
    auto recalcedClamp = quantParams.getMax()[0] - quantParams.getMin()[0];

    if (toNegative) {
        return mv::QuantizationParams({255}, {quantParams.getScale(0)}, {-recalcedClamp}, {0});
    } else {
        return mv::QuantizationParams({0}, {quantParams.getScale(0)}, {0}, {recalcedClamp});
    }
}

mv::Data::OpListIterator portDepthwise(mv::ComputationModel& model, mv::Data::TensorIterator inputTensor, mv::Data::OpListIterator task,
                        mv::QuantizationParams quantParams)
{
    mv::OpModel om(model);
    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = {255};
    std::vector<double> scale = {0.00392156862745098};
    std::vector<double> min = {min_inf};
    std::vector<double> max = {max_inf};
    auto inputShape = inputTensor->getShape();
    std::string name = task->getName();

    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    std::vector<int64_t> weightsData(inputShape[mv::IO_CHANNEL_DIMENSION], 0);
    weights = om.constantInt("",
                        weightsData,
                        {1, 1, inputShape[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order::getZMajorID(4));
    weights->setQuantParams(weightsQuantParams);
    auto depthwise_conv = om.depthwiseConv(name + "_PPEDepthwiseConv", inputTensor, weights, {1,1}, {0,0,0,0}, 1);
    depthwise_conv->setDType(mv::DType("UInt8"));
    depthwise_conv->setQuantParams(adjustFQtoHalfSum(quantParams, true));
    auto depthwise_convOp = om.getSourceOp(depthwise_conv);
    auto weightsOp = om.getSourceOp(weights);
    unsigned currentOpId = task->get<unsigned>("opId");
    weightsOp->set<unsigned>("opId", currentOpId);
    depthwise_convOp->set<unsigned>("opId", currentOpId);

    return depthwise_convOp;
}

mv::Data::OpListIterator portConv(mv::ComputationModel& model, mv::Data::OpListIterator task,
                mv::Data::OpListIterator previousOp, uint8_t branch)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Data::TensorIterator weightsTensor, biasTensor, inputTensor;
    inputTensor = previousOp->getInputTensor(0);
    bool hasBias = false;
    //NOTE: Add second input needs to be handled as well
    if (previousOp->getOpType() == "Conv" || previousOp->getOpType() == "DepthwiseConv")
    {
        weightsTensor = previousOp->getInputTensor(1);
        hasBias = previousOp->hasAttr("bias");
        if (hasBias)
            biasTensor = dm.getTensor(previousOp->get<std::string>("bias"));
    }

    auto kernelShape = weightsTensor->getShape();
    double alpha = task->get<double>("alpha");
    std::vector<int64_t> weightsData;

    std::vector<int64_t> zp;
    std::vector<double> scale;
    std::vector<double> min;
    std::vector<double> max;

    if (!weightsTensor->hasAttr("quantParams")) {
        weightsTensor->setQuantParams(mv::QuantizationParams::initial());
    }

    //Note: Quantization of weights
    if (branch == 0)
    {
        zp = weightsTensor->get<mv::QuantizationParams>("quantParams").getZeroPoint();
        scale = weightsTensor->get<mv::QuantizationParams>("quantParams").getScale();
        min = weightsTensor->get<mv::QuantizationParams>("quantParams").getMin();
        max = weightsTensor->get<mv::QuantizationParams>("quantParams").getMax();
        weightsData = weightsTensor->getIntData();
    }
    else
    {
        std::vector <int64_t> old_zp = weightsTensor->get<mv::QuantizationParams>("quantParams").getZeroPoint();
        std::vector <double> old_scale = weightsTensor->get<mv::QuantizationParams>("quantParams").getScale();
        std::vector <double> old_min = weightsTensor->get<mv::QuantizationParams>("quantParams").getMin();
        std::vector <double> old_max = weightsTensor->get<mv::QuantizationParams>("quantParams").getMax();

        if (old_scale.size() == 1)
        {
            old_zp = std::vector <int64_t>(kernelShape[mv::KERNEL_OUTPUT_CHANNELS], old_zp[0]);
            old_scale = std::vector <double>(kernelShape[mv::KERNEL_OUTPUT_CHANNELS], old_scale[0]);
            old_min = std::vector <double>(kernelShape[mv::KERNEL_OUTPUT_CHANNELS], old_min[0]);
            old_max = std::vector <double>(kernelShape[mv::KERNEL_OUTPUT_CHANNELS], old_max[0]);
            mv::QuantizationParams vectorQuant = {old_zp, old_scale, old_min, old_max};
            weightsTensor->set<mv::QuantizationParams>("quantParams", vectorQuant);
        } else {
            if (kernelShape[mv::KERNEL_OUTPUT_CHANNELS] != old_zp.size() ||
                kernelShape[mv::KERNEL_OUTPUT_CHANNELS] != old_scale.size() ||
                kernelShape[mv::KERNEL_OUTPUT_CHANNELS] != old_min.size() ||
                kernelShape[mv::KERNEL_OUTPUT_CHANNELS] != old_max.size())
                throw mv::OpError("PPEAccuracy", "Wrong convolution quantization");
        }

        zp.clear();
        scale.clear();
        min.clear();
        max.clear();
        for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
        {
            zp.push_back(255 - weightsTensor->get<mv::QuantizationParams>("quantParams").getZeroPoint()[k]);
            scale.push_back(alpha*weightsTensor->get<mv::QuantizationParams>("quantParams").getScale()[k]);
            min.push_back(weightsTensor->get<mv::QuantizationParams>("quantParams").getMin()[k]);
            max.push_back(weightsTensor->get<mv::QuantizationParams>("quantParams").getMax()[k]);

            for (size_t c = 0; c < kernelShape[mv::KERNEL_INPUT_CHANNELS]; c++)
            {
                for (size_t h = 0; h < kernelShape[mv::KERNEL_HEIGHT]; h++)
                {
                    for (size_t w = 0; w < kernelShape[mv::KERNEL_WIDTH]; w++)
                    {
                        auto currWeight = (int64_t)weightsTensor->at({w,h,c,k});
                        weightsData.push_back(255 - currWeight);
                    }
                }
            }
        }
    }

    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    mv::QuantizationParams quantParams = task->getOutputTensor(0)->getQuantParams();
    std::string constantName = previousOp->getInputTensor()[1]->getName() + std::to_string(branch);
    std::string name = task->getName();

    auto weights = om.constantInt(constantName,
                        weightsData,
                        {weightsTensor->getShape()[mv::KERNEL_WIDTH], weightsTensor->getShape()[mv::KERNEL_HEIGHT],
                        weightsTensor->getShape()[mv::KERNEL_INPUT_CHANNELS], weightsTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]},
                        mv::DType("UInt8"),
                        mv::Order("NCHW"));
    weights->setQuantParams(weightsQuantParams);

    auto conv = om.conv(name + std::to_string(branch) + "_PPEConv",
                        inputTensor, weights,
                        previousOp->get<std::array<unsigned short, 2>>("stride"),
                        previousOp->get<std::array<unsigned short, 4>>("padding"),
                        previousOp->get<unsigned>("dilationFactor"),
                        previousOp->get<unsigned>("group"));
    conv->setDType(mv::DType("UInt8"));
    conv->setQuantParams(adjustFQtoHalfSum(quantParams, false));
    auto convOp = om.getSourceOp(conv);

    if (hasBias)
    {
        if (branch == 0)
        {
            std::vector <double> biasScale;
            for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
            {
                biasScale.push_back(scale[k] * inputTensor->get<mv::QuantizationParams>("quantParams").getScale()[0]);
            }
            biasTensor->set<mv::QuantizationParams>("quantParams", {{0},{biasScale},{min_inf},{max_inf}});
            om.addAttr(convOp, "bias", biasTensor->getName());
        }
        else
        {
            double biasOldScale, real_bias;
            std::vector <int64_t> biasData;
            std::vector <double> biasScale;
            for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
            {
                biasOldScale = weightsTensor->get<mv::QuantizationParams>("quantParams").getScale()[k]
                        * inputTensor->get<mv::QuantizationParams>("quantParams").getScale()[0];
                real_bias = ((int64_t) biasTensor->at(k)) * biasOldScale;
                real_bias = -alpha * real_bias;
                auto newQuantizedValue = std::round(real_bias
                                    /(scale[k] * inputTensor->get<mv::QuantizationParams>("quantParams").getScale()[0]));
                if (newQuantizedValue > 2147483647)
                    newQuantizedValue = 2147483647;
                else if (newQuantizedValue < -2147483648)
                    newQuantizedValue = -2147483648;
                biasData.push_back(newQuantizedValue);
                biasScale.push_back(scale[k] * inputTensor->get<mv::QuantizationParams>("quantParams").getScale()[0]);
            }
            mv::QuantizationParams biasQuant = mv::QuantizationParams({{0},biasScale,
                                        {-min_inf},{max_inf}});

            mv::Data::TensorIterator branch2biasTensor;
            std::string branch2biasTensorName = mv::createBiasName(convOp->getName());
            branch2biasTensor = dm.defineTensor(mv::Tensor(branch2biasTensorName, biasTensor->getShape(),
                            biasTensor->getDType(), biasTensor->getOrder(), biasData, biasQuant));
            om.addAttr(convOp, "bias", branch2biasTensorName);
        }
    }

    auto weightsOp = om.getSourceOp(weights);
    unsigned currentOpId = task->get<unsigned>("opId");
    weightsOp->set<unsigned>("opId", currentOpId + branch);
    convOp->set<unsigned>("opId", currentOpId + ADD_INPUT_FLOWS + 1);

    return convOp;
}

mv::Data::OpListIterator portRelu(mv::ComputationModel& model, mv::Data::TensorIterator inputTensor, mv::Data::OpListIterator task, mv::QuantizationParams quantParams)
{
    mv::OpModel om(model);
    std::string name = task->getName();
    auto relu0 = om.relu(name + "RELU", inputTensor);
    relu0->setDType(mv::DType("UInt8"));
    relu0->setQuantParams(adjustFQtoHalfSum(quantParams, false));
    auto relu_op = om.getSourceOp(relu0);
    unsigned currentOpId = task->get<unsigned>("opId");
    relu_op->set<unsigned>("opId", currentOpId);
    return relu_op;
}

mv::Data::OpListIterator portAdd(mv::ComputationModel& model, std::vector <mv::Data::TensorIterator> inputs ,mv::Data::OpListIterator task, mv::QuantizationParams quantParams)
{
    mv::OpModel om(model);
    std::string name = task->getName();
    auto eltwise = om.eltwise(name + "PPEeltwise", inputs, "Add");
    eltwise->setDType(mv::DType("UInt8"));
    eltwise->setQuantParams(quantParams);
    auto eltwise_op = om.getSourceOp(eltwise);
    unsigned currentOpId = task->get<unsigned>("opId");
    eltwise_op->set<unsigned>("opId", currentOpId);
    return eltwise_op;
}

void provideAccuracyinPPEs(mv::ComputationModel& model)
{
    //NOTE: The idea of this workaround is that the ppe mechanism in hardware
    //seems not to round the negative values correctly, so we apply a workaround
    //with executing 2 parallel convolutions, one with positive values and the other
    //with the negatives and then adding the outputs, to avoid ppe rounding
    mv::OpModel om(model);
    mv::DataModel dm(model);

    if (!checkPPEAccuracy(model))
        return;
    auto leakyRelus = om.getOps("LeakyRelu");
    auto leakyRelu = leakyRelus.begin();

    while (leakyRelu != leakyRelus.end())
    {
        auto leakyReluOp = *leakyRelu;
        auto leakyInputTensor = leakyReluOp->getInputTensor(mv::IO_TENSOR_INPUT);
        auto leakyOutputTensor = leakyReluOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        auto parentOp = om.getSourceOp(leakyInputTensor);
        auto leakyReluQuantParams = leakyOutputTensor->getQuantParams();

        // Cannot fuse eltwise into the PPE LeakyReLU
        if (!parentOp->hasWeights()) {
            leakyRelu++;
            continue;
        }

        // skip fp16 ops
        if (leakyReluQuantParams.isNeutral()) {
            leakyRelu++;
            continue;
        }

        // activations should be per-tensor
        if (leakyReluQuantParams.getZeroPoint().size() != 1 ||
            leakyReluQuantParams.getScale().size() != 1 ||
            leakyReluQuantParams.getMin().size() != 1 ||
            leakyReluQuantParams.getMax().size() != 1) {
            leakyRelu++;
            continue;
        }

        // skip fp16 ops
        auto inputTensor = parentOp->getInputTensor(0);
        if (!inputTensor->isQuantized() || inputTensor->getQuantParams().isNeutral() ||
             inputTensor->getDType() != mv::DType("UInt8")) {
            leakyRelu++;
            continue;
        }

        // skip fp16 ops
        auto weightsTensor = parentOp->getInputTensor(1);
        if (!weightsTensor->isQuantized() || weightsTensor->getQuantParams().isNeutral() ||
             weightsTensor->getDType() != mv::DType("UInt8")) {
            leakyRelu++;
            continue;
        }

        uint8_t branch = 0;
        uint8_t branchConcat = 0;
        uint8_t branchEltwise = 0;
        uint8_t branchDWConv = 0;
        std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, leakyOutputTensor);
        //NOTE: For now take into account that only the first sink
        //goes to concat but in general this needs to be searched
        for (std::size_t sinkOperatorsId = 0;sinkOperatorsId < sinkOperators.size(); sinkOperatorsId++)
        {
            if (sinkOperators[sinkOperatorsId]->getOpType() == "Concat")
            {
                for (uint8_t inputId = 0; inputId < sinkOperators[sinkOperatorsId]->getInputTensor().size(); inputId++)
                {
                    if (sinkOperators[sinkOperatorsId]->getInputTensor()[inputId]->getName() == leakyOutputTensor->getName())
                        branchConcat = inputId;
                }
            }
            if (sinkOperators[sinkOperatorsId]->getOpType() == "Eltwise")
            {
                for (uint8_t inputId = 0; inputId < sinkOperators[sinkOperatorsId]->getInputTensor().size(); inputId++)
                {
                    if (sinkOperators[sinkOperatorsId]->getInputTensor()[inputId]->getName() == leakyOutputTensor->getName())
                        branchEltwise = inputId;
                }
            }
            if (sinkOperators[sinkOperatorsId]->getOpType() == "DepthwiseConv")
            {
                for (uint8_t inputId = 0; inputId < sinkOperators[sinkOperatorsId]->getInputTensor().size(); inputId++)
                {
                    if (sinkOperators[sinkOperatorsId]->getInputTensor()[inputId]->getName() == leakyOutputTensor->getName())
                        branchDWConv = inputId;
                }
            }
        }
        mv::Data::OpListIterator relu0, conv0, conv1, depthwise1, relu1;
        for (uint8_t i = 0; i < ADD_INPUT_FLOWS; i++)
        {
            if (i == 0)
            {
                conv0 = portConv(model,leakyReluOp, parentOp, i);
                relu0 = portRelu(model, conv0->getOutputTensor(0), conv0, leakyReluQuantParams);
            }
            else
            {
                conv1 = portConv(model, leakyReluOp, parentOp, i);
                relu1 = portRelu(model, conv1->getOutputTensor(0), conv1, leakyReluQuantParams);
                depthwise1 = portDepthwise(model, relu1->getOutputTensor(0), leakyReluOp, leakyReluQuantParams);
            }
        }
        std::vector<mv::Data::TensorIterator> inputs;
        inputs.push_back(relu0->getOutputTensor(0));
        inputs.push_back(depthwise1->getOutputTensor(0));
        auto add0 = portAdd(om, inputs, leakyReluOp, leakyReluQuantParams);
        auto backup = parentOp.leftmostOutput();
        om.undefineFlow(backup);
        if ((parentOp->getOpType() == "Conv") || (parentOp->getOpType() == "DepthwiseConv"))
            om.removeOp(om.getSourceOp(parentOp->getInputTensor(1)));
        om.removeOp(leakyReluOp);
        om.removeOp(parentOp);
        for (std::size_t numberOfSink = 0; numberOfSink < sinkOperators.size(); numberOfSink++)
        {
            if (sinkOperators[numberOfSink]->getOpType() == "Eltwise")
                branch = branchEltwise;
            if (sinkOperators[numberOfSink]->getOpType() == "DepthwiseConv")
                branch = branchDWConv;
            if (sinkOperators[numberOfSink]->getOpType() == "Concat")
                branch = branchConcat;
            sinkOperators[numberOfSink]->setInputTensor(add0->getOutputTensor(0), branch, false);
            om.defineFlow(add0->getOutputTensor(0), sinkOperators[numberOfSink], branch);
        }
        leakyRelu++;
    }
}
