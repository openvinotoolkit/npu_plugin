#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void alignTo16ChannelsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignWeightsTensor(mv::OpModel& om, const mv::Data::TensorIterator &weightsTensor, unsigned channelsPadded, size_t axis, std::string typeOp);
static void alignBiasTensor(mv::Data::OpListIterator &opIt, const mv::Data::TensorIterator biasTensor, unsigned biasTensorSizePadded, mv::DataModel dm);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignTo16Channels)
            .setFunc(alignTo16ChannelsFcn)
            .setDescription(
                "Aligns I/O channels involved in DPUTask to 16");
    }
}

//NOTE: Mark the Ops that do not have output channels aligned to 16,in serialization you align their dims
//and provide the appropriate Tensor for DMA
void alignTo16ChannelsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    int pad = globalConfigParams->hasAttr("VPU2ChannelPadding") ? globalConfigParams->get<int>("VPU2ChannelPadding") : 16;
    auto dpuTasks = om.getOps("DPUTask");

    for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        auto taskOp = opIt->get<std::string>("taskOp");
        auto outputTensor = opIt->getOutputTensor(0);
        auto outputTensorShape = outputTensor->getShape();
        auto outputTensorChannels = outputTensorShape[mv::IO_CHANNEL_DIMENSION];
        auto opStrategy = opIt->get<std::string>("splitStrategy");

        if (outputTensorChannels % pad != 0)
        {
            opIt->set<bool>("alignment", true);
            if (!outputTensor->hasAttr("alignment"))
            {
                outputTensor->set<bool>("alignment", true);
            }
        }
        auto outputChannelsPadded = mv::round_up(outputTensorShape[mv::IO_CHANNEL_DIMENSION], pad);

        if(taskOp == "Conv" || taskOp == "DepthWiseConv")
        {
            auto inputTensor = opIt->getInputTensor(0);
            if (!inputTensor->hasAttr("alignment") && inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % pad != 0)
            {
                inputTensor->set<bool>("alignment", true);
            }
        }

        if(taskOp == "Conv")
        {
            auto weightsTensor = opIt->getInputTensor(1);
            if(outputChannelsPadded != weightsTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS])
            {
                alignWeightsTensor(om, weightsTensor, outputChannelsPadded, mv::KERNEL_OUTPUT_CHANNELS, taskOp);
                weightsTensor = opIt->getInputTensor(1);
            }
            auto inputTensor = opIt->getInputTensor(0);
            //if the previous op will be alligned you have to align the weights of the current
            if (inputTensor->hasAttr("alignment"))
            {
                auto inputTensorChannels = inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto inputChannelsPadded = mv::round_up(inputTensorChannels, pad);
                alignWeightsTensor(om, weightsTensor, inputChannelsPadded, mv::KERNEL_INPUT_CHANNELS, taskOp);

            }
        }
        else if (taskOp == "DepthwiseConv")
        {
            auto weightsTensor = opIt->getInputTensor(1);
            if(outputChannelsPadded != weightsTensor->getShape()[mv::KERNEL_INPUT_CHANNELS])
            {
                alignWeightsTensor(om, weightsTensor, outputChannelsPadded, mv::KERNEL_INPUT_CHANNELS, taskOp);
                weightsTensor = opIt->getInputTensor(1);
            }
            auto inputTensor = opIt->getInputTensor(0);
            if (inputTensor->hasAttr("alignment"))
            {
                auto inputTensorChannels = inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto inputChannelsPadded = mv::round_up(inputTensorChannels, pad);
                alignWeightsTensor(om, weightsTensor, inputChannelsPadded, mv::KERNEL_INPUT_CHANNELS, taskOp);
            }
        }

        if(opIt->hasAttr("bias"))
        {
            auto biasTensorName = opIt->get<std::string>("bias");
            auto biasTensor = om.getTensor(biasTensorName);
            alignBiasTensor(opIt, biasTensor, outputChannelsPadded, dm);
        }
    }
}

static void alignWeightsTensor(mv::OpModel& om, const mv::Data::TensorIterator &weightsTensor, unsigned channelsPadded, size_t axis, std::string typeOp)
{
    auto weightsTensorShape = weightsTensor->getShape();
    auto weightsTensorOrder = weightsTensor->getOrder();
    auto weightsTensorDType = weightsTensor->getDType();
    auto weightsTensorWidth = weightsTensorShape[mv::KERNEL_WIDTH];
    auto weightsTensorHeight = weightsTensorShape[mv::KERNEL_HEIGHT];
    auto weightsTensorInputChannels = weightsTensorShape[mv::KERNEL_INPUT_CHANNELS];
    auto weightsTensorOutputChannels = weightsTensorShape[mv::KERNEL_OUTPUT_CHANNELS];

    auto newShape = mv::Shape({0, 0, 0, 0});
    if (typeOp == "Conv")
    {
        if ((axis == mv::KERNEL_INPUT_CHANNELS && weightsTensorInputChannels == channelsPadded) ||
            (axis == mv::KERNEL_OUTPUT_CHANNELS && weightsTensorOutputChannels == channelsPadded))
            return;
        newShape = mv::Shape({weightsTensorWidth, weightsTensorHeight, weightsTensorInputChannels, channelsPadded});
        if (axis == mv::KERNEL_INPUT_CHANNELS)
            newShape = mv::Shape({weightsTensorWidth, weightsTensorHeight, channelsPadded, weightsTensorOutputChannels});
    }
    else
        newShape = mv::Shape({weightsTensorWidth, weightsTensorHeight, channelsPadded, 1});

    int64_t zeroPoint = 0;
    mv::QuantizationParams weightsTensorQuantizationParams({},{},{},{});

    if(weightsTensor->isQuantized())
    {
        weightsTensorQuantizationParams = weightsTensor->get<mv::QuantizationParams>("quantParams");
        zeroPoint = weightsTensorQuantizationParams.getZeroPoint()[0];
    }

    auto newData = std::vector<mv::DataElement>(newShape.totalSize(), mv::DataElement(weightsTensorDType.isDoubleType(), zeroPoint));
    auto constantOp = om.getSourceOp(weightsTensor);
    auto outFlows = mv::getOutputDataFlow(om, constantOp, false);
    mv::Data::TensorIterator newKernel = om.constantDataElement(newData, newShape, weightsTensorDType, weightsTensorOrder, weightsTensorQuantizationParams, mv::createAlignConstantName(constantOp->getName()));

    //DO NOT CHANGE THE LIMITS OF THE LOOP! THERE IS A REASON WHY IT'S DONE LIKE THIS AND NOT USING THE AUXILIARY VARIABLES
    for(unsigned oc = 0; oc < weightsTensorShape[mv::KERNEL_OUTPUT_CHANNELS]; ++oc)
        for(unsigned ic = 0; ic < weightsTensorShape[mv::KERNEL_INPUT_CHANNELS]; ++ic)
            for(unsigned kw = 0; kw < weightsTensorShape[mv::KERNEL_WIDTH]; ++kw)
                for(unsigned kh = 0; kh < weightsTensorShape[mv::KERNEL_HEIGHT]; ++kh)
                    newKernel->at({kw,kh,ic,oc}) = weightsTensor->at({kw,kh,ic,oc});

    om.getSourceOp(newKernel)->set<unsigned>("opId", constantOp->get<unsigned>("opId"));

    om.removeOp(constantOp);
    mv::setOutputDataFlow(om, newKernel, outFlows);
}

static void alignBiasTensor(mv::Data::OpListIterator &opIt, const mv::Data::TensorIterator biasTensor, unsigned biasTensorSizePadded, mv::DataModel dm)
{
    //Bias case is easier since it is 1D
    auto biasTensorDType = biasTensor->getDType();
    auto biasTensorSize = biasTensor->getShape()[0];


    auto biasTensorName = opIt->get<std::string>("bias");
    if(biasTensorSizePadded != biasTensorSize)
    {
        auto biasTensorQuantizationParams = biasTensor->get<mv::QuantizationParams>("quantParams");
        int64_t zeroPoint = 0;
//        if(biasTensor->isQuantized())
//            zeroPoint = biasTensorQuantizationParams.getZeroPoint()[0];

        auto newData = std::vector<mv::DataElement>(biasTensorSizePadded, mv::DataElement(biasTensorDType.isDoubleType(), zeroPoint));
        auto newBiasTensor = dm.defineTensor(mv::createAlignConstantName(biasTensorName), {biasTensorSizePadded}, biasTensorDType, mv::Order("W"), newData);
        if(biasTensor->isQuantized())
            newBiasTensor->set<mv::QuantizationParams>("quantParams", biasTensorQuantizationParams);

        for(unsigned i = 0; i < biasTensorSize; ++i)
            newBiasTensor->at({i}) = biasTensor->at({i});

        dm.undefineTensor(biasTensorName);
        opIt->erase("bias");
        opIt->set<std::string>("bias", newBiasTensor->getName());

        //check for other ops with the same bias tensor, and upate teh attribute
        mv::OpModel om(dm);
        auto dpuTasks = om.getOps("DPUTask");
        for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
        {
            auto updateOpIt = *vecIt;
            if(updateOpIt->hasAttr("bias") && updateOpIt->get<std::string>("bias") == biasTensorName)
            {
                updateOpIt->erase("bias");
                updateOpIt->set<std::string>("bias", newBiasTensor->getName());
            }
        }
    }
}
