#include "include/fathom/computation/model/op_model.hpp"

mv::OpModel::OpModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::OpModel::OpModel(Logger &logger) :
ComputationModel(logger)
{

}

mv::OpModel::OpModel(const ComputationModel &other) :
ComputationModel(other)
{

}

bool mv::OpModel::defaultControlFlow_(DataContext::OpListIterator &op)
{
    ControlContext::OpListIterator currentOp = opsGraph_->get_second_iterator(op);
    ControlContext::FlowListIterator newFlow = controlGraph_.edge_insert(lastOp_, currentOp, allocator_.make_owner<ControlFlow>(logger_, lastOp_, currentOp));

    if (newFlow == controlFlowEnd_)
        return false;

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());
    lastOp_ = currentOp;

    return true;

}

bool mv::OpModel::defaultStage_(DataContext::OpListIterator &op)
{

    auto stageIt = addStage_();
    
    if (!addToStage_(stageIt, op))
        return false;

    return true;

}

mv::DataContext::OpListIterator mv::OpModel::switchContext(ControlContext::OpListIterator &other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::DataContext::OpListIterator mv::OpModel::input(const Shape &shape, DType dType, Order order, const string &name)
{

    input_ = dataGraph_.node_insert(allocator_.make_owner<Input>(logger_, shape, dType, order, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());
    lastOp_ = opsGraph_->get_second_iterator(input_);
    return input_;

}

mv::DataContext::OpListIterator mv::OpModel::output(DataContext::OpListIterator &input, const string &name)
{

    output_ = dataGraph_.node_insert(allocator_.make_owner<Output>(logger_, name));

    auto inputTensorIt = findUnpopulatedTensor_(input->getOutputName());
    if (inputTensorIt == unpopulatedTensorEnd_)
    {
        inputTensorIt = getUnpopulatedTensor_(input->getOutputDef());
    }

    output_->setInput(inputTensorIt, 0);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());
    
    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(input, output_, allocator_.make_owner<DataFlow>(logger_, input, output_, inputTensorIt));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(output_);

    return output_;

}

mv::DataContext::OpListIterator mv::OpModel::conv(DataContext::OpListIterator &input, DataContext::OpListIterator &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    DataContext::OpListIterator convIt = dataGraph_.node_insert(allocator_.make_owner<Conv>(logger_, strideX, strideY, padX, padY, name));
    
    auto inputTensorIt = findUnpopulatedTensor_(input->getOutputName());
    if (inputTensorIt == unpopulatedTensorEnd_)
    {
        inputTensorIt = getUnpopulatedTensor_(input->getOutputDef());
    }

    auto weightsTensorIt = findUnpopulatedTensor_(weights->getOutputName());
    if (weightsTensorIt == unpopulatedTensorEnd_)
    {
        weightsTensorIt = getUnpopulatedTensor_(weights->getOutputDef());
    }

    convIt->setInput(inputTensorIt, 0);
    convIt->setInput(weightsTensorIt, 1);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + convIt->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(input, convIt, allocator_.make_owner<DataFlow>(logger_, input, convIt, inputTensorIt));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator weightsFlow = dataGraph_.edge_insert(weights, convIt, allocator_.make_owner<DataFlow>(logger_, weights, convIt, weightsTensorIt));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + weightsFlow->toString());

    defaultControlFlow_(convIt);
    defaultStage_(convIt);

    return convIt;
}

mv::DataContext::OpListIterator mv::OpModel::maxpool(DataContext::OpListIterator &input, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    DataContext::OpListIterator poolIt = dataGraph_.node_insert(allocator_.make_owner<MaxPool>(logger_, kernelShape, strideX, strideY, padX, padY, name));
    
    auto inputTensorIt = findUnpopulatedTensor_(input->getOutputName());
    if (inputTensorIt == unpopulatedTensorEnd_)
    {
        inputTensorIt = getUnpopulatedTensor_(input->getOutputDef());
    }

    poolIt->setInput(inputTensorIt, 0);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + poolIt->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(input, poolIt, allocator_.make_owner<DataFlow>(logger_, input, poolIt, inputTensorIt));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(poolIt);
    defaultStage_(poolIt);

    return poolIt;
}

mv::DataContext::OpListIterator mv::OpModel::concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name)
{

    DataContext::OpListIterator concatIt = dataGraph_.node_insert(allocator_.make_owner<Concat>(logger_, name));
    
    auto input0TensorIt = findUnpopulatedTensor_(input0->getOutputName());
    if (input0TensorIt == unpopulatedTensorEnd_)
    {
        input0TensorIt = getUnpopulatedTensor_(input0->getOutputDef());
    }

    auto input1TensorIt = findUnpopulatedTensor_(input1->getOutputName());
    if (input1TensorIt == unpopulatedTensorEnd_)
    {
        input1TensorIt = getUnpopulatedTensor_(input1->getOutputDef());
    }

    concatIt->setInput(input0TensorIt, 0);
    concatIt->setInput(input1TensorIt, 1);
    
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + concatIt->toString());

    DataContext::FlowListIterator input0Flow = dataGraph_.edge_insert(input0, concatIt, allocator_.make_owner<DataFlow>(logger_, input0, concatIt, input0TensorIt));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input0Flow->toString());

    DataContext::FlowListIterator input1Flow = dataGraph_.edge_insert(input1, concatIt, allocator_.make_owner<DataFlow>(logger_, input1, concatIt, input1TensorIt));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input1Flow->toString());

    defaultControlFlow_(concatIt);
    defaultStage_(concatIt);

    return concatIt;

}

mv::DataContext::OpListIterator mv::OpModel::constant(const ConstantTensor &tensor, const string &name)
{
    DataContext::OpListIterator constIt = dataGraph_.node_insert(allocator_.make_owner<Constant>(logger_, tensor, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + constIt->toString());
    return constIt;
}

bool mv::OpModel::addAttr(DataContext::OpListIterator &opIt, const string &name, const Attribute &attr)
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


mv::GroupContext::MemberIterator mv::OpModel::addGroupElement(DataContext::OpListIterator &newElement, GroupContext::GroupIterator &group)
{

    allocator::owner_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);

}

bool mv::OpModel::removeGroupElement(DataContext::OpListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationOp> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::vector<mv::Shape> mv::OpModel::getInputShapes(DataContext::OpListIterator &op)
{

    vector<Shape> shapes;

    for (auto it = op.leftmostInput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

mv::vector<mv::Shape> mv::OpModel::getOutputShapes(DataContext::OpListIterator &op)
{

    vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}