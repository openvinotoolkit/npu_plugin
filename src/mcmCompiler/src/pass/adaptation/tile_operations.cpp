#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_quantization.hpp"
#include <cmath>

const size_t MAX_LIMIT_KERNEL = 11;
const size_t MID_LIMIT_KERNEL_H = 5;
const size_t MID_LIMIT_KERNEL_W = 5;
const size_t NUMBER_OF_PARTITIONS = 9;
const size_t GROUP_DILATION = 1;

static void tileOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void partitionOperation(mv::Data::OpListIterator& opIt, std::size_t kernel_w, std::size_t kernel_h,
                        mv::ComputationModel& model);
void padInputTensor(mv::Data::OpListIterator& opIt, mv::OpModel& model);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TileOps)
        .setFunc(tileOpsFcn)
        .setDescription(
            "Unfortunately HW supports only operations till 11 kernel size, \
                so replace/tile bigger kernels to smaller."
        );
    }
}

void tileOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model,
                       mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    auto convOps = om.getOps("Conv");

    for (auto conv: convOps) {

        assert(conv->inputSlots() == 2);

        mv::Data::TensorIterator weights = conv->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET);
        std::size_t kernel_size_w = weights->getShape()[mv::KERNEL_WIDTH];
        std::size_t kernel_size_h = weights->getShape()[mv::KERNEL_HEIGHT];

        if (kernel_size_w <= MAX_LIMIT_KERNEL && kernel_size_h <= MAX_LIMIT_KERNEL)
            continue;

        // 1. pad
        padInputTensor(conv, om);

        // 2. split
        partitionOperation(conv, kernel_size_w, kernel_size_h, model);

    }
}

void fillPadTensor(std::vector<mv::Data::TensorIterator>& concatedTensors,
                   mv::OpModel& om,
                   const size_t dataSize, const unsigned opId, const mv::Shape& newShape,
                   const mv::Data::TensorIterator& input)
{
    if (dataSize == 0)
        return;

    mv::Data::TensorIterator pad;

    if (input->isQuantized()) {
        std::vector<int64_t> newData(dataSize, input->getQuantParams().getZeroPoint(0));
        pad = om.constantInt("",
                                 newData,
                                 newShape,
                                 input->getDType(),
                                 input->getOrder());
        pad->setQuantParams(input->getQuantParams());
    }
    else {
        std::vector<double> newData(dataSize, 0);
        pad = om.constant("",
                         newData,
                         newShape,
                         input->getDType(),
                         input->getOrder());
    }

    om.getSourceOp(pad)->set<unsigned>("opId", opId);
    pad->set<bool>("is_pad", true);

    concatedTensors.emplace_back(pad);
}

void padInputTensor(mv::Data::OpListIterator& opIt, mv::OpModel& om)
{
    auto originalPadding = opIt->get<std::array<unsigned short, 4>>("padding");

    if (std::find_if(std::begin(originalPadding), std::end(originalPadding), [&](const int& pad){
       return pad != 0;
    }) == std::end(originalPadding))
        return;

    mv::Data::TensorIterator inputTensor = opIt->getInputTensor(mv::IO_TENSOR_INPUT);
    unsigned opId = opIt->get<unsigned>("opId");
    mv::Shape inputTensorShape = inputTensor->getShape();
    std::size_t otherDimSize = inputTensorShape[mv::IO_CHANNEL_DIMENSION] * inputTensorShape[mv::IO_BATCH_DIMENSION];

    // Create top/bottom padding, width of original tensor, height of t/b padding
    size_t topSize = inputTensorShape[mv::IO_WIDTH_DIMENSION] * originalPadding[mv::PADDING_TOP];
    size_t bottomSize = inputTensorShape[mv::IO_WIDTH_DIMENSION] * originalPadding[mv::PADDING_BOT];

    std::vector<mv::Data::TensorIterator> concatedTensors;

    fillPadTensor(concatedTensors, om, topSize * otherDimSize, opId,
                  {inputTensorShape[mv::IO_WIDTH_DIMENSION], originalPadding[mv::PADDING_TOP], inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]},
                  inputTensor);

    concatedTensors.emplace_back(inputTensor);
    fillPadTensor(concatedTensors, om, bottomSize * otherDimSize, opId,
                  {inputTensorShape[mv::IO_WIDTH_DIMENSION], originalPadding[mv::PADDING_BOT], inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]},
                  inputTensor);

    if (concatedTensors.size() > 1) {
        auto concatH = om.concat(opIt->getName() + "_padH", concatedTensors, "H");
        concatH->setDType(inputTensor->getDType());

        if(inputTensor->isQuantized())
           concatH->setQuantParams(inputTensor->getQuantParams());
        om.getSourceOp(concatH)->set<unsigned>("opId", opId);
        inputTensor = concatH;
    }

    concatedTensors.clear();

    size_t newHeight = inputTensorShape[mv::IO_HEIGHT_DIMENSION] + originalPadding[mv::PADDING_TOP] + originalPadding[mv::PADDING_BOT];
    // Create left/right padding, height of (original tensor+ t/b padding), width of l/r padding
    size_t leftSize = newHeight * originalPadding[mv::PADDING_LEFT];
    size_t rightSize = newHeight * originalPadding[mv::PADDING_RIGHT];

    fillPadTensor(concatedTensors, om, leftSize * otherDimSize, opId,
                  {originalPadding[mv::PADDING_LEFT], newHeight, inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]},
                  inputTensor);

    concatedTensors.emplace_back(inputTensor);
    fillPadTensor(concatedTensors, om, rightSize * otherDimSize, opId,
                  {originalPadding[mv::PADDING_RIGHT], newHeight, inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]},
                  inputTensor);

    if (concatedTensors.size() > 1) {
        auto concatW = om.concat(opIt->getName() + "_padW", concatedTensors, "W");
        concatW->setDType(inputTensor->getDType());

        if(inputTensor->isQuantized())
           concatW->setQuantParams(inputTensor->getQuantParams());
        om.getSourceOp(concatW)->set<unsigned>("opId", opId);
        inputTensor = concatW;
    }
    
    auto sourceFlow = opIt.leftmostInput();
    om.undefineFlow(sourceFlow);
    opIt->setInputTensor(inputTensor, mv::IO_TENSOR_INPUT, false);
    om.defineFlow(inputTensor, opIt, mv::IO_TENSOR_INPUT);

    opIt->set<std::array<unsigned short, 4>>("padding", {0,0,0,0});
}

void calculateMultipliedQuantization(mv::QuantizationParams& params, const mv::Data::TensorIterator& outputTensor)
{
    if (!outputTensor->isQuantized())
        return;

    // the value of multiplier (how much we lengthen fq output range)
    size_t factor = 1;
    // assuming level was 256
    size_t level = 256;

    mv::QuantizationParams param = outputTensor->getQuantParams();
    auto max = param.getMax();
    auto min = param.getMin();

    std::transform(min.begin(), min.end(), max.begin(), min.begin(),
                   [factor](double& min_val, double& max_val)
    {
        double new_min = min_val - factor*(max_val-min_val);
        max_val = max_val + factor*(max_val-min_val);
        return new_min;
    });

    std::vector<double> scales = {0};
    std::vector<int64_t> zpoints = {0};
    mv::calcZeroPointAndScalePerChannel(max, min, level, outputTensor->getDType(), scales, zpoints);
    params = mv::QuantizationParams(zpoints, scales, max, min);
}

void partitionOperation(mv::Data::OpListIterator& opIt, std::size_t kernelW, std::size_t kernelH,
                        mv::ComputationModel& model)
{
    // the main approach is to calculate the right slice for kernel and input tensor
    // in order to emulate large kernel calculation and summarize them using eltwise
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // check inputs
    assert(opIt->inputSlots() == 2 || opIt->outputSlots() == 1);

    mv::Data::TensorIterator inputTensor  = opIt->getInputTensor(mv::IO_TENSOR_INPUT);
    mv::Data::TensorIterator weightTensor = opIt->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET);
    mv::Data::TensorIterator outputTensor = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT);

    std::array<unsigned short, 2> initialStride  = opIt->get<std::array<unsigned short, 2>>("stride");
    std::array<unsigned short, 4> padding = opIt->get<std::array<unsigned short, 4>>("padding");
    unsigned initialOpId = opIt->get<unsigned>("opId");

    size_t newKernelW = std::min(kernelW, MAX_LIMIT_KERNEL);
    size_t newKernelH = std::min(kernelH, MAX_LIMIT_KERNEL);

    size_t countW = std::ceil(static_cast<float>(kernelW) / newKernelW);
    size_t countH = std::ceil(static_cast<float>(kernelH) / newKernelH);

    mv::Shape beginInputShape, branchInputSize, beginWeightShape, branchWeightSize;

    // inaccuracy occurs than calculations are in fixed point (because of the clamp after each convolution)
    // here we handle it by extension of fq range until mixed precision mode it fixed
    // (compilation error occurs)
    mv::QuantizationParams multipliedParams = mv::QuantizationParams::empty();
    calculateMultipliedQuantization(multipliedParams, outputTensor);

    std::stack<mv::Data::TensorIterator> operators;
    for (size_t indexW = 0; indexW < countW; ++indexW) {

        for (size_t indexH = 0; indexH < countH; ++indexH) {

            beginWeightShape = {indexW*newKernelW, indexH*newKernelH, 0, 0};

            size_t weightWidth = countW == 1 || indexW < countW - 1
                    ? newKernelW: kernelW - (indexW)*newKernelW;
            size_t weightHeight = countH == 1 || indexW < countH - 1
                    ? newKernelH: kernelH - (indexH)*newKernelH;

            size_t startW = indexW*newKernelW;
            size_t startH = indexH*newKernelH;
            beginInputShape = {startW,startH, 0, 0};

            branchWeightSize = {weightWidth, weightHeight,
                                weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS],
                                weightTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]};

            size_t branchWidth = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] - startW -
                    (kernelW - ((indexW)*newKernelW + weightWidth));
            size_t branchHeight = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION] - startH -
                    (kernelH - ((indexH)*newKernelH + weightHeight));
            branchInputSize = {branchWidth, branchHeight,
                               inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],
                               inputTensor->getShape()[mv::IO_BATCH_DIMENSION]};

            size_t order = countW*indexW + indexH;

            auto sliceInput = om.slice(opIt->getName() + "_slice_Input" + std::to_string(order),
                                       inputTensor,
                                       beginInputShape,
                                       branchInputSize);

            if (inputTensor->isQuantized())
               sliceInput->setQuantParams(inputTensor->getQuantParams());

            auto sliceWeight = om.slice(opIt->getName() + "_slice_Weight" + std::to_string(order),
                                       weightTensor,
                                       beginWeightShape,
                                       branchWeightSize);
            if (weightTensor->isQuantized())
               sliceWeight->setQuantParams(weightTensor->getQuantParams());

            mv::Data::TensorIterator conv = om.conv(opIt->getName() + std::to_string(order),
                           sliceInput,
                           sliceWeight,
                           initialStride,
                           padding,
                           GROUP_DILATION,
                           GROUP_DILATION);
            conv->setDType(inputTensor->getDType());

            if (outputTensor->isQuantized()) {
               conv->setQuantParams(multipliedParams);
            }

            om.getSourceOp(sliceInput)->set<unsigned>("opId", initialOpId);
            om.getSourceOp(sliceWeight)->set<unsigned>("opId", initialOpId);

            auto convOp = om.getSourceOp(conv);
            convOp->set<unsigned>("opId", initialOpId);

            if (!operators.empty()) {
                auto currentTensorIt = om.eltwise(opIt->getName() + "ADD_Partition" + std::to_string(order),
                {operators.top(), conv}, "Add");
                auto currentEltwise = om.getSourceOp(currentTensorIt);
                currentEltwise->set<unsigned>("opId", initialOpId);
                operators.pop();

                if (outputTensor->isQuantized())
                    currentTensorIt->setQuantParams(outputTensor->getQuantParams());
                operators.push(currentTensorIt);
            }
            else {
               // because of the inaccuracy which occurs if
               // we set bias on maxpool, move it to the first slice
                if (opIt->hasAttr("bias")) {
                    std::string biasName = mv::createBiasName(convOp->getName());
                    mv::Data::TensorIterator bias = dm.getTensor(opIt->get<std::string>("bias"));
                    mv::Data::TensorIterator newBias = dm.defineTensor(mv::Tensor(biasName, bias->getShape(),
                                                       inputTensor->getDType(), bias->getOrder(), bias->getData()));
                    if (bias->isQuantized())
                        newBias->setQuantParams(bias->getQuantParams());

                    om.addAttr(convOp, "bias", biasName);
                }
               operators.push(conv);
            }
        }
    }

    // check if we've split something
    assert(!operators.empty());

    auto currentOp = om.getSourceOp(operators.top());

    // add neutral maxpool here to set postop on it
    // because of the limitations for eltwise
    auto maxTensorIt = om.maxPool(currentOp->getName() + "_MaxPool",
                                  operators.top(), {1, 1}, {1, 1}, {0, 0, 0, 0}, false);
    maxTensorIt->setDType(outputTensor->getDType());
    auto maxOp = om.getSourceOp(maxTensorIt);
    maxOp->set<unsigned>("opId", initialOpId);

    if (outputTensor->isQuantized())
       maxTensorIt->setQuantParams(outputTensor->getQuantParams());

    if (opIt->hasAttr("postOpTypes"))
        maxOp->set<std::vector<std::string>>("postOpTypes",
                 opIt->get<std::vector<std::string>>("postOpTypes"));

    std::vector<mv::Data::OpListIterator> nextOps = mv::findSinkLayers(dm, outputTensor);
    for (auto& sinkOps: nextOps) {
        sinkOps->setInputTensor(maxTensorIt, mv::IO_TENSOR_INPUT, false);
        om.defineFlow(maxTensorIt, sinkOps, mv::IO_TENSOR_INPUT);
    }
    om.removeOp(opIt);

}
