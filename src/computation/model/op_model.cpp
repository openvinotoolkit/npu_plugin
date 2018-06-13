#include "include/fathom/computation/model/op_model.hpp"

mv::OpModel::OpModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::OpModel::OpModel(const ComputationModel& other) :
ComputationModel(other)
{

}

bool mv::OpModel::defaultControlFlow_(DataContext::OpListIterator& op)
{

    ControlContext::OpListIterator currentOp = opsGraph_->get_second_iterator(op);
    ControlContext::FlowListIterator newFlow = controlGraph_.edge_insert(lastOp_, currentOp, allocator_.make_owner<ControlFlow>(lastOp_, currentOp));

    if (newFlow == controlFlowEnd_)
        return false;

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());
    lastOp_ = currentOp;

    return true;

}

bool mv::OpModel::defaultStage_(DataContext::OpListIterator& op)
{

    auto stageIt = addStage_();
    
    if (!addToStage_(stageIt, op))
        return false;

    return true;

}

mv::DataContext::OpListIterator mv::OpModel::checkInputTensor_(DataContext::TensorIterator& inputTensor)
{

    if (inputTensor == tensorEnd_)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output op - unspecified input tensor" );
        return opEnd();
    }

    auto sourceIt = findSourceOp_(inputTensor);
    
    if (sourceIt == opEnd())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output op - tensor '" + inputTensor->getName() + 
            "' does not belong to the computation model");
        return opEnd();
    }

    return sourceIt;

}

mv::DataContext::OpListIterator mv::OpModel::switchContext(ControlContext::OpListIterator& other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::DataContext::TensorIterator mv::OpModel::input(const Shape& shape, DType dType, Order order, const string& name)
{

    input_ = dataGraph_.node_insert(allocator_.make_owner<Input>(shape, dType, order, name));
    auto outputTensor = defineOutputTensor_(input_, 0);
    input_->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());

    lastOp_ = opsGraph_->get_second_iterator(input_);
    return outputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::output(DataContext::TensorIterator inputTensor, const string& name)
{
    
    auto sourceIt = checkInputTensor_(inputTensor);
    if (sourceIt == opEnd())
        return DataContext::TensorIterator();

    output_ = dataGraph_.node_insert(allocator_.make_owner<Output>(name));
    output_->setInput(inputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());
    
    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(sourceIt, output_, allocator_.make_owner<DataFlow>(sourceIt, output_, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(output_);

    return inputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name)
{
    DataContext::OpListIterator constantIt = dataGraph_.node_insert(allocator_.make_owner<Constant>(data, shape, dType, order, name));
    auto outputTensor = defineOutputTensor_(constantIt, 0);
    constantIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + constantIt->toString());
    return outputTensor;
}

mv::DataContext::TensorIterator mv::OpModel::constant(float_type *data, size_type size, const Shape& shape, DType dType, Order order, const string& name)
{
    dynamic_vector<float_type> dataVec(data, size);
    return constant(dataVec, shape, dType, order, name);
}

mv::DataContext::TensorIterator mv::OpModel::conv2D(DataContext::TensorIterator inputTensor, DataContext::TensorIterator filtersTensor, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{

    auto inputSourceIt = checkInputTensor_(inputTensor);
    if (inputSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto filtersSourceIt = checkInputTensor_(filtersTensor);
    if (filtersSourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator convIt = dataGraph_.node_insert(allocator_.make_owner<Conv2D>(stride, padding, name));
    convIt->setInput(inputTensor, 0);
    convIt->setInput(filtersTensor, 1);
    auto outputTensor = defineOutputTensor_(convIt, 0);
    convIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + convIt->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(inputSourceIt, convIt, allocator_.make_owner<DataFlow>(inputSourceIt, convIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator filtersFlow = dataGraph_.edge_insert(filtersSourceIt, convIt, allocator_.make_owner<DataFlow>(filtersSourceIt, convIt, filtersTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + filtersFlow->toString());

    defaultControlFlow_(convIt);
    defaultStage_(convIt);

    return outputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::fullyConnected(DataContext::TensorIterator inputTensor, DataContext::TensorIterator weightsTensor, const string& name)
{
    auto inputSourceIt = checkInputTensor_(inputTensor);
    if (inputSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto weightsSourceIt = checkInputTensor_(weightsTensor);
    if (weightsSourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator fullyConnectedIt = dataGraph_.node_insert(allocator_.make_owner<FullyConnected>(name));
    fullyConnectedIt->setInput(inputTensor, 0);
    fullyConnectedIt->setInput(weightsTensor, 1);
    auto outputTensor = defineOutputTensor_(fullyConnectedIt, 0);
    fullyConnectedIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + fullyConnectedIt->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(inputSourceIt, fullyConnectedIt, allocator_.make_owner<DataFlow>(inputSourceIt, fullyConnectedIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator filtersFlow = dataGraph_.edge_insert(weightsSourceIt, fullyConnectedIt, allocator_.make_owner<DataFlow>(weightsSourceIt, fullyConnectedIt, weightsTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + filtersFlow->toString());

    defaultControlFlow_(fullyConnectedIt);
    defaultStage_(fullyConnectedIt);

    return outputTensor;
}

mv::DataContext::TensorIterator mv::OpModel::maxpool2D(DataContext::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{

    auto sourceIt = checkInputTensor_(inputTensor);
    if (sourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator poolIt = dataGraph_.node_insert(allocator_.make_owner<MaxPool2D>(kernelSize, stride, padding, name));
    poolIt->setInput(inputTensor, 0);
    auto outputTensor = defineOutputTensor_(poolIt, 0);
    poolIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + poolIt->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(sourceIt, poolIt, allocator_.make_owner<DataFlow>(sourceIt, poolIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(poolIt);
    defaultStage_(poolIt);

    return outputTensor;
}

mv::DataContext::TensorIterator mv::OpModel::concat(DataContext::TensorIterator input0Tensor, DataContext::TensorIterator input1Tensor, const string& name)
{

    auto input0SourceIt = checkInputTensor_(input0Tensor);
    if (input0SourceIt == opEnd())
        return DataContext::TensorIterator();

    auto input1SourceIt = checkInputTensor_(input1Tensor);
    if (input1SourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator concatIt = dataGraph_.node_insert(allocator_.make_owner<Concat>(name));
    concatIt->setInput(input0Tensor, 0);
    concatIt->setInput(input1Tensor, 1);
    auto outputTensor = defineOutputTensor_(concatIt, 0);
    concatIt->setOutput(outputTensor, 0);
    
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + concatIt->toString());

    DataContext::FlowListIterator input0Flow = dataGraph_.edge_insert(input0SourceIt, concatIt, allocator_.make_owner<DataFlow>(input0SourceIt, concatIt, input0Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input0Flow->toString());

    DataContext::FlowListIterator input1Flow = dataGraph_.edge_insert(input1SourceIt, concatIt, allocator_.make_owner<DataFlow>(input1SourceIt, concatIt, input1Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input1Flow->toString());

    defaultControlFlow_(concatIt);
    defaultStage_(concatIt);

    return outputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::batchNorm(DataContext::TensorIterator inputTensor, DataContext::TensorIterator meanTensor, DataContext::TensorIterator varianceTensor, DataContext::TensorIterator offsetTensor, DataContext::TensorIterator scaleTensor, float_type varianceEps, const string& name)
{
    auto inputSourceIt = checkInputTensor_(inputTensor);
    if (inputSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto meanSourceIt = checkInputTensor_(meanTensor);
    if (meanSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto varianceSourceIt = checkInputTensor_(varianceTensor);
    if (varianceSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto offsetSourceIt = checkInputTensor_(offsetTensor);
    if (offsetSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto scaleSourceIt = checkInputTensor_(scaleTensor);
    if (scaleSourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator batchNormIt = dataGraph_.node_insert(allocator_.make_owner<BatchNorm>(varianceEps, name));
    batchNormIt->setInput(inputTensor, 0);
    batchNormIt->setInput(meanTensor, 1);
    batchNormIt->setInput(varianceTensor, 2);
    batchNormIt->setInput(offsetTensor, 3);
    batchNormIt->setInput(scaleTensor, 4);
    auto outputTensor = defineOutputTensor_(batchNormIt, 0);
    batchNormIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + batchNormIt->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(inputSourceIt, batchNormIt, allocator_.make_owner<DataFlow>(inputSourceIt, batchNormIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator meanFlow = dataGraph_.edge_insert(meanSourceIt, batchNormIt, allocator_.make_owner<DataFlow>(meanSourceIt, batchNormIt, meanTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + meanFlow->toString());

    DataContext::FlowListIterator varianceFlow = dataGraph_.edge_insert(varianceSourceIt, batchNormIt, allocator_.make_owner<DataFlow>(varianceSourceIt, batchNormIt, varianceTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + varianceFlow->toString());

    DataContext::FlowListIterator offsetFlow = dataGraph_.edge_insert(offsetSourceIt, batchNormIt, allocator_.make_owner<DataFlow>(offsetSourceIt, batchNormIt, offsetTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + offsetFlow->toString());

    DataContext::FlowListIterator scaleFlow = dataGraph_.edge_insert(scaleSourceIt, batchNormIt, allocator_.make_owner<DataFlow>(scaleSourceIt, batchNormIt, scaleTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + scaleFlow->toString());

    defaultControlFlow_(batchNormIt);
    defaultStage_(batchNormIt);

    return outputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::scale(DataContext::TensorIterator inputTensor, DataContext::TensorIterator scaleTensor, const string& name)
{

    auto inputSourceIt = checkInputTensor_(inputTensor);
    if (inputSourceIt == opEnd())
        return DataContext::TensorIterator();

    auto scaleSourceIt = checkInputTensor_(scaleTensor);
    if (scaleSourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator scaleIt = dataGraph_.node_insert(allocator_.make_owner<Scale>(name));
    scaleIt->setInput(inputTensor, 0);
    scaleIt->setInput(scaleTensor, 1);
    auto outputTensor = defineOutputTensor_(scaleIt, 0);
    scaleIt->setOutput(outputTensor, 0);
    
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + scaleIt->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(inputSourceIt, scaleIt, allocator_.make_owner<DataFlow>(inputSourceIt, scaleIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator scaleFlow = dataGraph_.edge_insert(scaleSourceIt, scaleIt, allocator_.make_owner<DataFlow>(scaleSourceIt, scaleIt, scaleTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + scaleFlow->toString());

    defaultControlFlow_(scaleIt);
    defaultStage_(scaleIt);

    return outputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::relu(DataContext::TensorIterator inputTensor, const string& name)
{

    auto sourceIt = checkInputTensor_(inputTensor);
    if (sourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator reluIt = dataGraph_.node_insert(allocator_.make_owner<ReLu>(name));
    reluIt->setInput(inputTensor, 0);
    auto outputTensor = defineOutputTensor_(reluIt, 0);
    reluIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + reluIt->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(sourceIt, reluIt, allocator_.make_owner<DataFlow>(sourceIt, reluIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(reluIt);
    defaultStage_(reluIt);

    return outputTensor;

}

mv::DataContext::TensorIterator mv::OpModel::add(DataContext::TensorIterator input0Tensor, DataContext::TensorIterator input1Tensor, const string& name)
{

    auto input0SourceIt = checkInputTensor_(input0Tensor);
    if (input0SourceIt == opEnd())
        return DataContext::TensorIterator();

    auto input1SourceIt = checkInputTensor_(input1Tensor);
    if (input1SourceIt == opEnd())
        return DataContext::TensorIterator();

    DataContext::OpListIterator sumIt = dataGraph_.node_insert(allocator_.make_owner<Add>(name));
    sumIt->setInput(input0Tensor, 0);
    sumIt->setInput(input1Tensor, 1);
    auto outputTensor = defineOutputTensor_(sumIt, 0);
    sumIt->setOutput(outputTensor, 0);
    
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + sumIt->toString());

    DataContext::FlowListIterator input0Flow = dataGraph_.edge_insert(input0SourceIt, sumIt, allocator_.make_owner<DataFlow>(input0SourceIt, sumIt, input0Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input0Flow->toString());

    DataContext::FlowListIterator input1Flow = dataGraph_.edge_insert(input1SourceIt, sumIt, allocator_.make_owner<DataFlow>(input1SourceIt, sumIt, input1Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input1Flow->toString());

    defaultControlFlow_(sumIt);
    defaultStage_(sumIt);

    return outputTensor;

}

mv::DataContext::OpListIterator mv::OpModel::getSourceOp(DataContext::TensorIterator tensor)
{
    return findSourceOp_(tensor);
}

bool mv::OpModel::addAttr(DataContext::OpListIterator opIt, const string& name, const Attribute& attr)
{

    return opIt->addAttr(name, attr);

}

bool mv::OpModel::isValid() const
{
    return ComputationModel::isValid();
}

mv::DataContext::OpListIterator mv::OpModel::getInput()
{
    return input_;
}

mv::DataContext::OpListIterator mv::OpModel::getOutput()
{
    return output_;
}

mv::DataContext::OpListIterator mv::OpModel::opEnd()
{
    return dataOpEnd_;
}


mv::GroupContext::MemberIterator mv::OpModel::addGroupElement(DataContext::OpListIterator newElement, GroupContext::GroupIterator group)
{

    allocator::owner_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);

}

bool mv::OpModel::removeGroupElement(DataContext::OpListIterator element, GroupContext::GroupIterator group)
{
    allocator::owner_ptr<ComputationOp> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::dynamic_vector<mv::Shape> mv::OpModel::getInputShapes(DataContext::OpListIterator& op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostInput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

mv::dynamic_vector<mv::Shape> mv::OpModel::getOutputShapes(DataContext::OpListIterator& op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}