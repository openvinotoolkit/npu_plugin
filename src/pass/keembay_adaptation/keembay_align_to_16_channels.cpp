#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void alignTo16ChannelsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

// ASSUMPTION: This pass is executed after ConvertOpsToTasks pass. This means that we have dputasks but no weights table nor sparsity map yet
// This is not a problem as these structures will be aligned to 16 automatically on their creation.
void alignTo16ChannelsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    int pad = globalConfigParams->hasAttr("VPU2ChannelPadding") ? globalConfigParams->get<int>("VPU2ChannelPadding") : 16;

    auto dpuTasks = om.getOps("DPUTask");
    for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
    {
        auto opIt = *vecIt;

        auto taskOp = opIt->get<std::string>("taskOp");
        if(taskOp != "ChannelMajorConvolution")
        {
            auto inputTensor = opIt->getInputTensor(0);
            auto outputTensor = opIt->getOutputTensor(0);

            auto inputTensorShape = inputTensor->getShape();
            auto outputTensorShape = outputTensor->getShape();

            auto inputChannelsPadded = mv::round_up(inputTensorShape[mv::IO_CHANNEL_DIMENSION], pad);
            auto outputChannelsPadded = mv::round_up(outputTensorShape[mv::IO_CHANNEL_DIMENSION], pad);

            auto newInputTensorShape = mv::Shape({inputTensorShape[mv::IO_WIDTH_DIMENSION], inputTensorShape[mv::IO_HEIGHT_DIMENSION], inputChannelsPadded, inputTensorShape[mv::IO_BATCH_DIMENSION]});
            auto newOutputTensorShape = mv::Shape({outputTensorShape[mv::IO_WIDTH_DIMENSION], outputTensorShape[mv::IO_HEIGHT_DIMENSION], outputChannelsPadded, outputTensorShape[mv::IO_BATCH_DIMENSION]});

            inputTensor->setShape(newInputTensorShape);
            outputTensor->setShape(newOutputTensorShape);

            if(taskOp == "Add" || taskOp == "Subtract" || taskOp == "Multiply")
            {
                auto inputTensor2 = opIt->getInputTensor(1);
                auto inputTensor2Shape = inputTensor2->getShape();
                auto inputChannels2Padded = mv::round_up(inputTensor2Shape[mv::IO_CHANNEL_DIMENSION], pad);
                auto newInputTensor2Shape = mv::Shape({inputTensorShape[mv::IO_WIDTH_DIMENSION], inputTensorShape[mv::IO_HEIGHT_DIMENSION], inputChannels2Padded, inputTensorShape[mv::IO_BATCH_DIMENSION]});
                inputTensor2->setShape(newInputTensor2Shape);
            }
            else if(opIt->hasAttr("hasWeights") && opIt->get<bool>("hasWeights"))
            {
                // For populated tensor this is not as easy as simply change the shape. We have to create a new operation.
                // Luckly, we have the utilities to do so :)
                auto weightsTensor = opIt->getInputTensor(1);
                auto weightsTensorShape = weightsTensor->getShape();

                auto weightsTensorInputChannels = weightsTensorShape[mv::KERNEL_INPUT_CHANNELS];
                auto weightsTensorInputChannelsPadded = mv::round_up(weightsTensorInputChannels, pad);

                auto weightsTensorOutputChannels = weightsTensorShape[mv::KERNEL_OUTPUT_CHANNELS];
                if(taskOp == "DepthwiseConv")
                    weightsTensorOutputChannels = weightsTensorShape[mv::KERNEL_INPUT_CHANNELS];
                auto weightsTensorOutputChannelsPadded = mv::round_up(weightsTensorOutputChannels, pad);

                if(weightsTensorInputChannelsPadded != weightsTensorInputChannels ||
                  weightsTensorOutputChannelsPadded != weightsTensorOutputChannels)
                {
                    auto weightsTensorOrder = weightsTensor->getOrder();
                    auto weightsTensorDType = weightsTensor->getDType();
                    auto weightsTensorWidth = weightsTensorShape[mv::KERNEL_WIDTH];
                    auto weightsTensorHeight = weightsTensorShape[mv::KERNEL_HEIGHT];


                    mv::Shape newShape = mv::Shape({weightsTensorWidth, weightsTensorHeight, weightsTensorInputChannelsPadded, weightsTensorOutputChannelsPadded});
                    if(taskOp == "DepthwiseConv")
                        newShape = mv::Shape({weightsTensorWidth, weightsTensorHeight, weightsTensorInputChannelsPadded, 1});
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
            }
            if(opIt->hasAttr("bias"))
            {
                //Bias case is easier since it is 1D
                auto biasTensorName = opIt->get<std::string>("bias");
                auto biasTensor = om.getTensor(biasTensorName);

                auto biasTensorDType = biasTensor->getDType();
                auto biasTensorSize = biasTensor->getShape()[0];
                auto biasTensorSizePadded = mv::round_up(biasTensorSize, 16);

                if(biasTensorSizePadded != biasTensorSize)
                {
                    auto biasTensorQuantizationParams = biasTensor->get<mv::QuantizationParams>("quantParams");
                    int64_t zeroPoint = 0;
                    if(biasTensor->isQuantized())
                        zeroPoint = biasTensorQuantizationParams.getZeroPoint()[0];

                    auto newData = std::vector<mv::DataElement>(biasTensorSizePadded, mv::DataElement(biasTensorDType.isDoubleType(), zeroPoint));
                    auto newBiasTensor = dm.defineTensor(mv::createAlignConstantName(biasTensorName), {biasTensorSizePadded}, biasTensorDType, mv::Order("W"), newData);
                    if(biasTensor->isQuantized())
                        newBiasTensor->set<mv::QuantizationParams>("quantParams", biasTensorQuantizationParams);

                    for(unsigned i = 0; i < biasTensorSize; ++i)
                        newBiasTensor->at({i}) = biasTensor->at({i});

                    dm.undefineTensor(biasTensorName);
                    opIt->erase("bias");
                    opIt->set<std::string>("bias", newBiasTensor->getName());
                }
            }
        }
    }
}
