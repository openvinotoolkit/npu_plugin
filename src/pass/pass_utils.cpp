#include "include/mcm/pass/pass_utils.hpp"

std::vector<std::pair<mv::Data::OpListIterator,size_t>> mv::getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp)
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;

    for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
    {
        auto consumer = output.sink();
        auto slot = output->get<size_t>("sinkInput");
        toReturn.push_back(std::make_pair(consumer, slot));
    }

    if(deleteOp)
    {
        auto backup = opIt;
        ++opIt;
        om.removeOp(backup);
    }

    return toReturn;
}

void mv::setOutputDataFlow(mv::OpModel& om, mv::Data::TensorIterator &dpuTaskOutputTensor, const std::vector<std::pair<mv::Data::OpListIterator,size_t>>& outDataFlows)
{
    for(auto& flowPair: outDataFlows)
    {
        flowPair.first->setInputTensor(dpuTaskOutputTensor, flowPair.second, false);
        om.defineFlow(dpuTaskOutputTensor, flowPair.first, flowPair.second);
    }
}

std::vector<mv::Control::OpListIterator> mv::getOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator opIt)
{
    std::vector<mv::Control::OpListIterator> toReturn;

    for(auto outputFlow = opIt.leftmostChild(); outputFlow != cm.opEnd(); ++outputFlow)
        toReturn.push_back(outputFlow);
    return toReturn;
}

std::vector<mv::Control::OpListIterator> mv::getInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator opIt)
{
    std::vector<mv::Control::OpListIterator> toReturn;

    for(auto outputFlow = opIt.leftmostParent(); outputFlow != cm.opEnd(); ++outputFlow)
        toReturn.push_back(outputFlow);
    return toReturn;
}

void mv::setInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& inputControlFlows)
{
    for(auto& inputOp: inputControlFlows)
        cm.defineFlow(inputOp, op);
}

void mv::setOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& outputControlFlows)
{
    for(auto& outputOp: outputControlFlows)
        cm.defineFlow(op, outputOp);
}

mv::Data::OpListIterator mv::linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt,
                                                 mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt"
            || paramOp->getOpType() == "ConstantDataElement")
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    if(sourceTensor == om.tensorEnd())
        sourceTensor = parentOpIt->getOutputTensor(0);

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

void calcZeroPointAndScalePerTensor(double outputMax,  double outputMin, double& outScale, int64_t& outZp)
{
    outScale = (outputMax - outputMin)/255;
    if (outputMin >= 0.0)
        outZp = 0;
    else if (outputMax <= 0.0)
        outZp = 255;
    else if ((outputMin < 0.0) && (outputMax > 0.0))
    {
        auto max_diff = (outputMax/(std::abs(outputMin) + outputMax)) * 255;
        outZp = std::ceil(255 - max_diff);
    }
}

void updateInfMinMaxPerTensor(mv::Data::TensorIterator tensor)
{
    auto& tensorQuantization = tensor->get<mv::QuantizationParams>("quantParams");

    //Note: if input Tensor has min, max of infs...we need to compute them
    if (tensorQuantization.infinitelimits())
    {
        //Quantization equation Real = scale(Quantized - zeroPoint)
        double maximumFloat = tensorQuantization.getScale()[0] * (255 - tensorQuantization.getZeroPoint()[0]);
        double minimumFloat = -tensorQuantization.getZeroPoint()[0] * tensorQuantization.getScale()[0];
        if (minimumFloat == -0)
            minimumFloat = 0;

        mv::QuantizationParams newTensorQuantization(tensorQuantization.getZeroPoint(),
                                                    tensorQuantization.getScale(),{minimumFloat},{maximumFloat});
        tensor->set<mv::QuantizationParams>("quantParams", newTensorQuantization);
    }
}

void updateInfMinMaxPerChannel(mv::Data::TensorIterator tensor)
{
    auto& tensorQuantization = tensor->get<mv::QuantizationParams>("quantParams");

    //Note: Do not care if populated or unpopulated....batch = 1
    if (tensorQuantization.infinitelimits())
    {
        std::vector <double> maximums, minimums;
        double maximumFloat, minimumFloat;
        for (uint32_t channel = 0; channel < tensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]; channel++)
        {
            //Quantization equation Real = scale(Quantized - zeroPoint)
            maximumFloat = tensorQuantization.getScale()[channel] * (255 - tensorQuantization.getZeroPoint()[0]);
            minimumFloat = -tensorQuantization.getZeroPoint()[0] * tensorQuantization.getScale()[channel];
            if (minimumFloat == -0)
                minimumFloat = 0;
            maximums.push_back(maximumFloat);
            minimums.push_back(minimumFloat);
        }
        mv::QuantizationParams newTensorQuantization(tensorQuantization.getZeroPoint(),
                                                    tensorQuantization.getScale(),minimums, maximums);
        tensor->set<mv::QuantizationParams>("quantParams", newTensorQuantization);
    }
}
