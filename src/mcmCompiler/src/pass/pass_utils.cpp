#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

std::vector<std::pair<mv::Data::OpListIterator,size_t>> mv::getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp)
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;
    auto outputTensor = opIt->getOutputTensor()[0];

    for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
    {
        auto consumer = output.sink();
        std::size_t slot = 0;
        for (std::size_t input_idx = 0; input_idx < consumer->getInputTensor().size(); input_idx++)
            if (consumer->getInputTensor()[input_idx]->getName() == outputTensor->getName())
                slot = input_idx;
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

/// Note: This helper can't remove a paramOp shared by two sinkOps
void mv::removeConstantOp(mv::OpModel & om, mv::Data::OpListIterator paramOp){
    mv::DataModel dm(om);
    if (findSinkLayers(dm, paramOp->getOutputTensor(mv::IO_TENSOR_OUTPUT)).size() == 1){
        om.removeOp(paramOp);
    }
}

void mv::removeOperation(mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        /// It's a pity that opIter not support postfix increment
        auto backup= paramOp;
        ++paramOp;
        
        if (backup->getOutputTensor(0) != sourceTensor &&
            (backup->getOpType() == "Constant" ||
            backup->getOpType() == "ConstantInt" ||
            backup->getOpType() == "ConstantDataElement"))
        {
            removeConstantOp(om, backup);
        }
    }

    om.removeOp(opIt);
}

mv::Data::OpListIterator mv::linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt,
                                                 mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
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
        auto backup= paramOp;
        ++paramOp;
        
        if (backup->getOutputTensor(0) != sourceTensor && (backup->getOpType() == "Constant" || backup->getOpType() == "ConstantInt"
            || backup->getOpType() == "ConstantDataElement"))
        {
            removeConstantOp(om, backup);
        }
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
mv::Data::OpListIterator mv::linkNewOperationsReplacement(mv::Data::OpListIterator parentOpIt,
                                                      mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
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
        auto backup= paramOp;
        ++paramOp;
        
        if (backup->getOutputTensor(0) != sourceTensor && (backup->getOpType() == "Constant" || backup->getOpType() == "ConstantInt"
            || backup->getOpType() == "ConstantDataElement"))
        {
            removeConstantOp(om, backup);
        }
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


mv::Data::OpListIterator mv::linkNewMultipleOperationsReplacement(mv::Data::OpListIterator parentOpIt,
                                                      std::vector<mv::Data::TensorIterator> sourceTensors, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    //consumers
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    for (auto sourceTensorIt = sourceTensors.begin(); sourceTensorIt != sourceTensors.end(); sourceTensorIt++)
    {
        while(paramOp != om.opEnd())
        {
            auto backup= paramOp;
            ++paramOp;
        
            if (backup->getOutputTensor(0) != *sourceTensorIt && (backup->getOpType() == "Constant" || backup->getOpType() == "ConstantInt"
                || backup->getOpType() == "ConstantDataElement"))
            {
                removeConstantOp(om, backup);
            }
        }
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (auto sourceTensorIt = sourceTensors.begin(); sourceTensorIt != sourceTensors.end(); sourceTensorIt++)
    {
        if(*sourceTensorIt == om.tensorEnd())
            *sourceTensorIt = parentOpIt->getOutputTensor(0);

        for (unsigned j = 0; j < opsToLink.size(); ++j)
        {
            opsToLink[j]->setInputTensor(*sourceTensorIt, inputSlots[j], false);
            om.defineFlow(*sourceTensorIt, opsToLink[j], inputSlots[j]);
        }
    }

    return opIt;
}

mv::Data::OpListIterator mv::linkNewMultipleOperationsReplacementRemoveFlows(mv::Data::OpListIterator parentOpIt,
                                                      std::vector<mv::Data::TensorIterator> sourceTensors, mv::OpModel & om, mv::Data::OpListIterator opIt)
{

    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    //consumers
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    for (auto sinkFlow = opIt.leftmostInput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        if (std::find(sourceTensors.begin(), sourceTensors.end(), sinkFlow->getTensor()) != sourceTensors.end())
        {
            auto op = sinkFlow.source();
            if (op->getOpType() == "Constant" ||
                op->getOpType() == "ConstantInt" ||
                op->getOpType() == "ConstantDataElement") {
                auto flowToRemove = sinkFlow;
                ++sinkFlow;
                om.undefineFlow(flowToRemove);
            }
        }
        else
            ++sinkFlow;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (auto sourceTensorIt = sourceTensors.begin(); sourceTensorIt != sourceTensors.end(); sourceTensorIt++)
    {
        if(*sourceTensorIt == om.tensorEnd())
            *sourceTensorIt = parentOpIt->getOutputTensor(0);

        for (unsigned j = 0; j < opsToLink.size(); ++j)
        {
            opsToLink[j]->setInputTensor(*sourceTensorIt, inputSlots[j], false);
            om.defineFlow(*sourceTensorIt, opsToLink[j], inputSlots[j]);
        }
    }

    return opIt;
}

mv::Data::OpListIterator mv::linkNewOperationsReplacementRemoveFlows(mv::Data::OpListIterator childOpIt,
                                                      mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;
    //consumers
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        flowsToRemove.push_back(sinkFlow);
    }

    for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
    {
        om.undefineFlow(flowsToRemove[flowIdx]);
    }
    om.removeOp(opIt);
    opIt = childOpIt;

    //create links
    for(unsigned op = 0 ; op < opsToLink.size(); ++op)
    {
        opsToLink[op]->setInputTensor(sourceTensor, inputSlots[op], false);
        om.defineFlow(sourceTensor, opsToLink[op], inputSlots[op]);
    }
    return opIt;
}

mv::Data::TensorIterator mv::insertDMAReplacementRemoveFlows(mv::OpModel& om, mv::Data::OpListIterator opIt,
    mv::Data::TensorIterator input, mv::DmaDirection const& direction, int8_t const &port,
    std::vector<mv::Data::FlowListIterator> flows, std::vector<std::size_t> inSlots,
    std::vector<mv::Data::OpListIterator> sinks, std::string const& dmaOpName)
{
    mv::DataModel dm(om);
    auto dmaTaskOut = om.dMATask(dmaOpName, input, direction, port);
    dmaTaskOut->setQuantParams(input->getQuantParams());
    auto dmaTaskOp = om.getSourceOp(dmaTaskOut);

    dmaTaskOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));

    for(std::size_t idx = 0; idx < flows.size(); ++idx)
    {
        auto sink = sinks.at(idx);
        auto slot = inSlots.at(idx);
        om.undefineFlow(flows.at(idx));
        sink->setInputTensor(dmaTaskOut, slot, false);
        om.defineFlow(dmaTaskOp, 0, sink, slot);
    }

    return dmaTaskOut;
}

mv::Data::TensorIterator mv::dequantizeWeightsToFP16(
    mv::Data::TensorIterator tensor,
    mv::Data::OpListIterator childOp,
    mv::OpModel &om)
{
    // For both fast access by avoiding data transposing
    // and ease of data interation set order to channel major
    // of arbitrary dimensions
    auto tensorShape = tensor->getShape();
    auto backupOrder = tensor->getOrder();
    tensor->setOrder(Order::getColMajorID(tensorShape.ndims()));

    auto numChannels = childOp->getOpType() == "DepthwiseConv" ||
        (childOp->getOpType() == "DPUTask" &&
        childOp->get<std::string>("taskOp") == "DepthwiseConv") ?
        tensorShape[mv::KERNEL_INPUT_CHANNELS] :
        tensorShape[mv::KERNEL_OUTPUT_CHANNELS];
    auto wSetSize = tensorShape.totalSize() / numChannels;

    auto quantTensorData = tensor->getIntData();
    auto quantParams = tensor->getQuantParams();

    // Step 1 dequantized the weights data
    auto dequantTensorData = std::vector<double>(quantTensorData.size());
    for(std::size_t chIdx = 0; chIdx < numChannels; chIdx++)
    {
        auto chScale = quantParams.getScale(chIdx);
        auto chZp = quantParams.getZeroPoint(chIdx);
        std::transform(
            quantTensorData.cbegin() + chIdx * wSetSize,
            quantTensorData.cbegin() + (chIdx + 1) * wSetSize,
            dequantTensorData.begin() + chIdx * wSetSize,
            [chScale, chZp] (const int64_t &intVal)
            {return static_cast<double>(intVal - chZp) * chScale;});
    }

    // Step 2 explicit conversion to fp16
    auto dequantFP16TensorData = std::vector<int64_t>(dequantTensorData.size());
    std::transform(
        dequantTensorData.cbegin(),
        dequantTensorData.cend(),
        dequantFP16TensorData.begin(),
        [] (const double &floatVal)
        {return mv::fp32_to_fp16(floatVal);});

    // Step 3 create new constantInt op due to FP16 internal representation
    // and link correctly
    auto sourceIntOp = om.getSourceOp(tensor);
    auto attrsToCopy = tensor->getAttrs(
        {"dType", "Shape", "order", "sourceOp", "flows", "quantParams"});
    auto dequantFP16Weights = om.constantInt(
        sourceIntOp->getName() + "_dequantFP16",
        dequantFP16TensorData,
        tensor->getShape(),
        mv::DType("Float16"),
        tensor->getOrder()
    );
    om.getSourceOp(dequantFP16Weights)->set<unsigned>("opId",
        sourceIntOp->get<unsigned>("opId"));
    dequantFP16Weights->setAttrs(attrsToCopy);
    dequantFP16Weights->setQuantParams(mv::QuantizationParams::initial());

    dequantFP16Weights->setOrder(backupOrder);
    om.getSourceOp(dequantFP16Weights)->set<mv::Order>("order", backupOrder);

    return dequantFP16Weights;
}

//template <class T>
//std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName)
//{
//    if (value.size() == 1)
//        return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

//    // We enter in this case if and only if we specified multi channel scales and
//    // the tensor has been aligned
//    if (value.size() < size)
//    {
//        auto toReturn = mv::utils::generateSequence<T>(size, static_cast<T>(0) , 0);
//        for(unsigned i = 0; i < value.size(); ++i)
//            toReturn[i] = value[i];
//        return toReturn;
//    }

//    if (value.size() == size)
//        return value;

//    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
//                std::to_string(value.size()));
//}

std::vector<double> extendToK(size_t size, std::vector<double> value, std::string tensorName)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<double>(size, static_cast<double>(value[0]) , 0);

    // We enter in this case if and only if we specified multi channel scales and
    // the tensor has been aligned
    if (value.size() < size)
    {
        auto toReturn = mv::utils::generateSequence<double>(size, static_cast<double>(0) , 0);
        for(unsigned i = 0; i < value.size(); ++i)
            toReturn[i] = value[i];
        return toReturn;
    }

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}

std::vector<int64_t> extendToK(size_t size, std::vector<int64_t> value, std::string tensorName)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<int64_t>(size, static_cast<int64_t>(value[0]) , 0);

    // We enter in this case if and only if we specified multi channel scales and
    // the tensor has been aligned
    if (value.size() < size)
    {
        auto toReturn = mv::utils::generateSequence<int64_t>(size, static_cast<int64_t>(0) , 0);
        for(unsigned i = 0; i < value.size(); ++i)
            toReturn[i] = value[i];
        return toReturn;
    }

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}

std::vector<mv::Data::OpListIterator> mv::findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    if ((tensor)->hasAttr("flows"))
    {
        auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
        for(auto flowName : flowsNames)
        {
            auto df = dataModel.getDataFlow(flowName);
            sinkOperations.push_back(df.sink());
        }
    }
    return sinkOperations;
}

bool mv::checkA0Sparsity(const mv::OpModel& model)
{
    return model.hasGlobalConfigParam("enableSparsityA0") && model.getGlobalConfigParam("enableSparsityA0").get<bool>();
}

bool mv::checkA0SOHSparsityBug(mv::Data::FlowListIterator flow, std::string referenceDevice, mv::Target target)
{
    if (target != mv::Target::ma2490 || referenceDevice != "A0")
        return false;
    auto sink = flow.sink();
    auto tensor = flow->getTensor();

    if(!tensor->isPopulated())
    {
        if(sink->hasAttr("splitStrategy"))
        {
            std::string splitStrategy = sink->get<std::string>("splitStrategy");

            if(splitStrategy == "SplitOverH" &&
               sink->getOpType() == "DPUTask" &&
               sink->get<std::string>("taskOp") == "Conv" &&
               (
                sink->get<std::array<unsigned short, 2>>("kSize")[1] > 1))

                return true;

        }

    }
    return false;
}


bool mv::isVectorsEqual(const std::vector<double>& left, const std::vector<double>& right) {
    if(left.size() != right.size()) {
        return false;
    }

    for (unsigned i = 0; i < left.size(); i++) {
        //if (fabs(left[i] - right[i]) > std::numeric_limits<float>::epsilon()) {
        if (fabs(left[i] - right[i]) > (1.0e-4) ) {
            return  false;
        }
    }
    return true;
}

bool mv::isEqualScale(const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
    //in keembay the two eltwises can have different zero point
    bool isScaleEqual = isVectorsEqual(left.getScale(), right.getScale());
    return isScaleEqual;
}

bool mv::isEqual(const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
    bool isZpEqual = left.getZeroPoint() == right.getZeroPoint();
    bool isMinEqual = isVectorsEqual(left.getMin(), right.getMin());
    bool isMaxEqual = isVectorsEqual(left.getMax(), right.getMax());
    bool isScaleEqual = isVectorsEqual(left.getScale(), right.getScale());
    return isZpEqual && isMinEqual && isMaxEqual && isScaleEqual;
}

// Need to enable PPEAccuracy when the option is true or when the model is SuperResolution
bool mv::checkPPEAccuracy(mv::ComputationModel& model) {
    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    bool PPEAccuracy = globalParams->hasAttr("PPEAccuracy") ? globalParams->get<bool>("PPEAccuracy") : false;

    // WA for SuperResolution enabling
    // Hardcoded by input number and shape
    size_t inputNumber = om.getNumNetworkInputs();
    if (inputNumber == 3) {
        auto input0 = om.getNetworkInputs()[0];
        if (input0->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] == 192)
            PPEAccuracy = true;
    }
    return PPEAccuracy;
}

std::vector<std::string>::const_iterator mv::findIsDPUPwlPostOp(const std::vector<std::string>& postOps, const mv::TargetDescriptor& td) {
    for (auto itr = postOps.begin(); itr != postOps.end(); ++itr) {
        if(td.isDpuPwl(*itr)) {
            return itr;
        }
    }
    return postOps.end();
}

bool mv::matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto opIt = it;

    for (auto& layer : pattern) {
        if (opIt->getOpType() != layer) {
            return false;
        }

        opIt = om.getSourceOp(opIt->getInputTensor(0));
    }

    return true;
}

bool mv::matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::Data::OpListIterator& lastIt, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto opIt = it;

    for (auto& layer : pattern) {
        if (opIt->getOpType() != layer) {
            return false;
        }

        lastIt = opIt;
        opIt = om.getSourceOp(opIt->getInputTensor(0));
    }

    return true;
}
