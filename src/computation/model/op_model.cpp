#include "include/fathom/computation/model/op_model.hpp"

mv::OpModel::OpModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::OpModel::OpModel(const ComputationModel &other) :
ComputationModel(other)
{

}

bool mv::OpModel::defaultControlFlow_(DataContext::OpListIterator &op)
{
    ControlContext::OpListIterator currentOp = opsGraph_->get_second_iterator(op);
    ControlContext::FlowListIterator newFlow = controlGraph_.edge_insert(lastOp_, currentOp, allocator_.make_owner<ControlFlow>(lastOp_, currentOp));

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

    input_ = dataGraph_.node_insert(allocator_.make_owner<Input>(shape, dType, order, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());
    lastOp_ = opsGraph_->get_second_iterator(input_);
    return input_;

}

mv::DataContext::OpListIterator mv::OpModel::output(DataContext::OpListIterator &inputIt, const string &name)
{

    output_ = dataGraph_.node_insert(allocator_.make_owner<Output>(name));

    auto inputTensor = findTensor_(inputIt->getOutputName());
    if (inputTensor == tensorEnd_)
    {
        inputTensor = getTensor_(inputIt->getOutputDef());
    }

    output_->setInput(inputTensor, 0);
    inputIt->setOutput(inputTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());
    
    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(inputIt, output_, allocator_.make_owner<DataFlow>(inputIt, output_, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(output_);

    return output_;

}

mv::DataContext::OpListIterator mv::OpModel::conv2D(DataContext::OpListIterator &inputIt, DataContext::OpListIterator &filtersIt, UnsignedVector2D stride, UnsignedVector4D padding, const string &name)
{

    DataContext::OpListIterator convIt = dataGraph_.node_insert(allocator_.make_owner<Conv>(stride, padding, name));
    
    auto inputTensor = findTensor_(inputIt->getOutputName());
    if (inputTensor == tensorEnd_)
    {
        inputTensor = getTensor_(inputIt->getOutputDef());
    }

    auto filtersTensor = findTensor_(filtersIt->getOutputName());
    if (filtersTensor == tensorEnd_)
    {
        filtersTensor = getTensor_(filtersIt->getOutputDef());
    }

    convIt->setInput(inputTensor, 0);
    inputIt->setOutput(inputTensor);
    convIt->setInput(filtersTensor, 1);
    filtersIt->setOutput(filtersTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + convIt->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(inputIt, convIt, allocator_.make_owner<DataFlow>(inputIt, convIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator filtersFlow = dataGraph_.edge_insert(filtersIt, convIt, allocator_.make_owner<DataFlow>(filtersIt, convIt, filtersTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + filtersFlow->toString());

    defaultControlFlow_(convIt);
    defaultStage_(convIt);

    return convIt;

}

mv::DataContext::OpListIterator mv::OpModel::maxpool2D(DataContext::OpListIterator &inputIt, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name)
{

    DataContext::OpListIterator poolIt = dataGraph_.node_insert(allocator_.make_owner<MaxPool>(kernelSize, stride, padding, name));
    
    auto inputTensor = findTensor_(inputIt->getOutputName());
    if (inputTensor == tensorEnd_)
    {
        inputTensor = getTensor_(inputIt->getOutputDef());
    }

    poolIt->setInput(inputTensor, 0);
    inputIt->setOutput(inputTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + poolIt->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(inputIt, poolIt, allocator_.make_owner<DataFlow>(inputIt, poolIt, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(poolIt);
    defaultStage_(poolIt);

    return poolIt;
}

mv::DataContext::OpListIterator mv::OpModel::concat(DataContext::OpListIterator &input0It, DataContext::OpListIterator &input1It, const string &name)
{

    DataContext::OpListIterator concatIt = dataGraph_.node_insert(allocator_.make_owner<Concat>(name));
    
    auto input0Tensor = findTensor_(input0It->getOutputName());
    if (input0Tensor == tensorEnd_)
    {
        input0Tensor = getTensor_(input0It->getOutputDef());
    }

    auto input1Tensor = findTensor_(input1It->getOutputName());
    if (input1Tensor == tensorEnd_)
    {
        input1Tensor = getTensor_(input1It->getOutputDef());
    }

    concatIt->setInput(input0Tensor, 0);
    input0It->setOutput(input0Tensor);
    concatIt->setInput(input1Tensor, 1);
    input1It->setOutput(input1Tensor);
    
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + concatIt->toString());

    DataContext::FlowListIterator input0Flow = dataGraph_.edge_insert(input0It, concatIt, allocator_.make_owner<DataFlow>(input0It, concatIt, input0Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input0Flow->toString());

    DataContext::FlowListIterator input1Flow = dataGraph_.edge_insert(input1It, concatIt, allocator_.make_owner<DataFlow>(input1It, concatIt, input1Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input1Flow->toString());

    defaultControlFlow_(concatIt);
    defaultStage_(concatIt);

    return concatIt;

}

mv::DataContext::OpListIterator mv::OpModel::constant(const dynamic_vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name)
{
    DataContext::OpListIterator constantIt = dataGraph_.node_insert(allocator_.make_owner<Constant>(data, shape, dType, order, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + constantIt->toString());
    return constantIt;
}

mv::DataContext::OpListIterator mv::OpModel::constant(float_type *data, size_type size, const Shape &shape, DType dType, Order order, const string &name)
{
    dynamic_vector<float_type> dataVec(data, size);
    return constant(dataVec, shape, dType, order, name);
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

mv::dynamic_vector<mv::Shape> mv::OpModel::getInputShapes(DataContext::OpListIterator &op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostInput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

mv::dynamic_vector<mv::Shape> mv::OpModel::getOutputShapes(DataContext::OpListIterator &op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}