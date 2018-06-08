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

mv::DataContext::OpListIterator mv::OpModel::output(DataContext::OpListIterator &input, const string &name)
{

    output_ = dataGraph_.node_insert(allocator_.make_owner<Output>(name));

    auto inputTensor = findTensor_(input->getOutputName());
    if (inputTensor == tensorEnd_)
    {
        inputTensor = getTensor_(input->getOutputDef());
    }

    output_->setInput(inputTensor, 0);
    input->setOutput(inputTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());
    
    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(input, output_, allocator_.make_owner<DataFlow>(input, output_, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(output_);

    return output_;

}

mv::DataContext::OpListIterator mv::OpModel::conv(DataContext::OpListIterator &input, DataContext::OpListIterator &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    DataContext::OpListIterator conv = dataGraph_.node_insert(allocator_.make_owner<Conv>(strideX, strideY, padX, padY, name));
    
    auto inputTensor = findTensor_(input->getOutputName());
    if (inputTensor == tensorEnd_)
    {
        inputTensor = getTensor_(input->getOutputDef());
    }

    auto weightsTensor = findTensor_(weights->getOutputName());
    if (weightsTensor == tensorEnd_)
    {
        weightsTensor = getTensor_(weights->getOutputDef());
    }

    conv->setInput(inputTensor, 0);
    input->setOutput(inputTensor);
    conv->setInput(weightsTensor, 1);
    weights->setOutput(weightsTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + conv->toString());

    DataContext::FlowListIterator inputFlow = dataGraph_.edge_insert(input, conv, allocator_.make_owner<DataFlow>(input, conv, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());

    DataContext::FlowListIterator weightsFlow = dataGraph_.edge_insert(weights, conv, allocator_.make_owner<DataFlow>(weights, conv, weightsTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + weightsFlow->toString());

    defaultControlFlow_(conv);
    defaultStage_(conv);

    return conv;
}

mv::DataContext::OpListIterator mv::OpModel::maxpool(DataContext::OpListIterator &input, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    DataContext::OpListIterator pool = dataGraph_.node_insert(allocator_.make_owner<MaxPool>(kernelShape, strideX, strideY, padX, padY, name));
    
    auto inputTensor = findTensor_(input->getOutputName());
    if (inputTensor == tensorEnd_)
    {
        inputTensor = getTensor_(input->getOutputDef());
    }

    pool->setInput(inputTensor, 0);
    input->setOutput(inputTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + pool->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(input, pool, allocator_.make_owner<DataFlow>(input, pool, inputTensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(pool);
    defaultStage_(pool);

    return pool;
}

mv::DataContext::OpListIterator mv::OpModel::concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name)
{

    DataContext::OpListIterator concat = dataGraph_.node_insert(allocator_.make_owner<Concat>(name));
    
    auto input0Tensor = findTensor_(input0->getOutputName());
    if (input0Tensor == tensorEnd_)
    {
        input0Tensor = getTensor_(input0->getOutputDef());
    }

    auto input1Tensor = findTensor_(input1->getOutputName());
    if (input1Tensor == tensorEnd_)
    {
        input1Tensor = getTensor_(input1->getOutputDef());
    }

    concat->setInput(input0Tensor, 0);
    input0->setOutput(input0Tensor);
    concat->setInput(input1Tensor, 1);
    input1->setOutput(input1Tensor);
    
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + concat->toString());

    DataContext::FlowListIterator input0Flow = dataGraph_.edge_insert(input0, concat, allocator_.make_owner<DataFlow>(input0, concat, input0Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input0Flow->toString());

    DataContext::FlowListIterator input1Flow = dataGraph_.edge_insert(input1, concat, allocator_.make_owner<DataFlow>(input1, concat, input1Tensor));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input1Flow->toString());

    defaultControlFlow_(concat);
    defaultStage_(concat);

    return concat;

}

mv::DataContext::OpListIterator mv::OpModel::constant(const vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name)
{
    DataContext::OpListIterator constant = dataGraph_.node_insert(allocator_.make_owner<Constant>(data, shape, dType, order, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + constant->toString());
    return constant;
}

mv::DataContext::OpListIterator mv::OpModel::constant(float_type *data, size_type size, const Shape &shape, DType dType, Order order, const string &name)
{
    vector<float_type> dataVec(data, size);
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