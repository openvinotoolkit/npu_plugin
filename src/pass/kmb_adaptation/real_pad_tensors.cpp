#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/shape.hpp"

static void alignUnpopulatedTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignPopulatedTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignWeightsTensor(mv::OpModel& om, const mv::Data::TensorIterator &weightsTensor, mv::Shape alignedShape);
static void alignBiasTensor(mv::Data::OpListIterator &opIt, const mv::Data::TensorIterator biasTensor, unsigned biasTensorSizePadded, mv::DataModel dm);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignUnpopulatedTensors)
            .setFunc(alignUnpopulatedTensors);

        MV_REGISTER_PASS(AlignPopulatedTensors)
            .setFunc(alignPopulatedTensors)
            .setDescription(
                "Aligns I/O channels involved in DPUTask to 16");
    }
}

// We can't use the registry at this point
// Since we only change activation tensor

// NOTE: What happens with slices?
void propagateShapeChange(mv::OpModel& om, const std::string& flowStr)
{
    auto flow = om.getDataFlow(flowStr);
    auto sink = flow.sink();

    std::string opType = sink->getOpType();

    if(opType == "DPUTask")
        opType = sink->get<std::string>("taskOp");

    if(opType == "Add" || opType == "Subtract" || opType == "Multiply" ||
       opType == "DepthwiseConv" || opType == "MaxPool")
    {
        auto inputTensor = flow->getTensor();
        auto inputShape = inputTensor->getShape();

        auto outputTensor = sink->getOutputTensor(0);
        auto outputShape = outputTensor->getShape();

        // If for whatever reason we pass through this tensor more than once, we
        // don't want to overwrite the original dimensions
        if(!outputTensor->hasAttr("oldDimensions"))
            outputTensor->set<mv::Shape>("oldDimensions", outputTensor->getShape());

        outputTensor->setShape({outputShape[mv::IO_WIDTH_DIMENSION], outputShape[mv::IO_HEIGHT_DIMENSION], inputShape[mv::IO_CHANNEL_DIMENSION], outputShape[mv::IO_BATCH_DIMENSION]});
        auto flows = outputTensor->get<std::set<std::string>>("flows");
        for(auto& flowStri: flows)
            propagateShapeChange(om, flowStri);
    }
}

//NOTE: Mark the Ops that do not have output channels aligned to 16,in serialization you align their dims
//and provide the appropriate Tensor for DMA
void alignUnpopulatedTensors(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto globalConfigParams = model.getGlobalConfigParams();
    int pad = globalConfigParams->hasAttr("VPU2ChannelPadding") ? globalConfigParams->get<int>("VPU2ChannelPadding") : 16;

    auto dpuTasks = om.topologicalSort();
    for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        if(opIt->getOpType() != "DPUTask")
            continue;

        auto outputTensor = opIt->getOutputTensor(0);
        auto outputTensorShape = outputTensor->getShape();
        auto outputTensorChannels = outputTensorShape[mv::IO_CHANNEL_DIMENSION];
        auto opStrategy = opIt->get<std::string>("splitStrategy");

        if (outputTensorChannels % pad != 0)
        {
            opIt->set<bool>("alignment", true);
            outputTensor->set<bool>("alignment", true);

            auto outputChannelsPadded = mv::round_up(outputTensorChannels, pad);

            // If for whatever reason we pass through this tensor more than once, we
            // don't want to overwrite the original dimensions
            if(!outputTensor->hasAttr("oldDimensions"))
                outputTensor->set<mv::Shape>("oldDimensions", outputTensor->getShape());

            outputTensor->setShape(mv::Shape({outputTensorShape[mv::IO_WIDTH_DIMENSION], outputTensorShape[mv::IO_HEIGHT_DIMENSION],
                                              outputChannelsPadded, outputTensorShape[mv::IO_BATCH_DIMENSION]}));

            auto flows = outputTensor->get<std::set<std::string>>("flows");
            for(auto& flowStr: flows)
                propagateShapeChange(om, flowStr);
        }
    }
}

void alignPopulatedTensors(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);


    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        if (layer->hasAttr("hasWeights") && layer->get<bool>("hasWeights"))
        {
            auto inputTensor = layer->getInputTensor(0);
            auto weightsTensor = layer->getInputTensor(1);
            auto outputTensor = layer->getOutputTensor(0);
            auto weightsTensorShape = weightsTensor->getShape();
            auto inputTensorShape = inputTensor->getShape();
            auto outputTensorShape = outputTensor->getShape();

            mv::Shape alignedShape;
            //NOTE: only Convs have weights (=) alignment
            if (layer->get<bool>("alignment"))
            {
                std::size_t outputChannelsPadded = outputTensorShape[mv::IO_CHANNEL_DIMENSION];

                if (layer->getOpType() == "Conv")
                {
                    alignedShape = mv::Shape({weightsTensorShape[mv::KERNEL_WIDTH], weightsTensorShape[mv::KERNEL_HEIGHT],
                                                       inputTensorShape[mv::IO_CHANNEL_DIMENSION], outputTensorShape[mv::IO_CHANNEL_DIMENSION]});
                }
                else if (layer->getOpType() == "DepthwiseConv")
                {
                    alignedShape = mv::Shape({weightsTensorShape[mv::KERNEL_WIDTH], weightsTensorShape[mv::KERNEL_HEIGHT],
                                                             inputTensorShape[mv::IO_CHANNEL_DIMENSION], 1});
                    outputChannelsPadded = inputTensorShape[mv::IO_CHANNEL_DIMENSION];
                }
                alignWeightsTensor(om, weightsTensor, alignedShape);
                if(layer->hasAttr("bias"))
                {
                    auto biasTensorName = layer->get<std::string>("bias");
                    auto biasTensor = om.getTensor(biasTensorName);
                    alignBiasTensor(layer, biasTensor, outputChannelsPadded, dm);
                }
            }
        }
    }

}

static void alignWeightsTensor(mv::OpModel& om, const mv::Data::TensorIterator &weightsTensor, mv::Shape alignedShape)
{
    auto weightsTensorOrder = weightsTensor->getOrder();
    auto weightsTensorDType = weightsTensor->getDType();
    int64_t zeroPoint = 0;
    mv::QuantizationParams weightsTensorQuantizationParams({},{},{},{});

    if(weightsTensor->isQuantized())
    {
        weightsTensorQuantizationParams = weightsTensor->get<mv::QuantizationParams>("quantParams");
        zeroPoint = weightsTensorQuantizationParams.getZeroPoint()[0];
    }

    auto newData = std::vector<mv::DataElement>(alignedShape.totalSize(), mv::DataElement(weightsTensorDType.isDoubleType(), zeroPoint));
    auto constantOp = om.getSourceOp(weightsTensor);
    auto outFlows = mv::getOutputDataFlow(om, constantOp, false);
    mv::Data::TensorIterator newKernel = om.constantDataElement(newData, alignedShape, weightsTensorDType, weightsTensorOrder, weightsTensorQuantizationParams, mv::createAlignConstantName(constantOp->getName()));

    //DO NOT CHANGE THE LIMITS OF THE LOOP! THERE IS A REASON WHY IT'S DONE LIKE THIS AND NOT USING THE AUXILIARY VARIABLES
    for(unsigned oc = 0; oc < alignedShape[mv::KERNEL_OUTPUT_CHANNELS]; ++oc)
        for(unsigned ic = 0; ic < alignedShape[mv::KERNEL_INPUT_CHANNELS]; ++ic)
            for(unsigned kw = 0; kw < alignedShape[mv::KERNEL_WIDTH]; ++kw)
                for(unsigned kh = 0; kh < alignedShape[mv::KERNEL_HEIGHT]; ++kh)
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

        //check for other ops with the same bias tensor, and upate teh attribute
        mv::OpModel om(dm);
        auto dpuTasks = om.getOps("DPUTask");
        for(auto layer = dpuTasks.begin(); layer != dpuTasks.end(); ++layer)
        {
            auto updateOpIt = *layer;
            if(updateOpIt->hasAttr("bias") && updateOpIt->get<std::string>("bias") == biasTensorName)
            {
                updateOpIt->erase("bias");
                updateOpIt->set<std::string>("bias", newBiasTensor->getName());
            }
        }
    }
}
