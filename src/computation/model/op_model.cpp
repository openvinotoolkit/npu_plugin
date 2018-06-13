#include "include/mcm/computation/model/op_model.hpp"

mv::OpModel::OpModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::OpModel::OpModel(const ComputationModel& other) :
ComputationModel(other)
{

}

bool mv::OpModel::defaultControlFlow_(Data::OpListIterator op)
{

    Control::OpListIterator currentOp = opsGraph_->get_second_iterator(op);
    Control::FlowListIterator newFlow = controlGraph_.edge_insert(lastOp_, currentOp, allocator_.make_owner<ControlFlow>(lastOp_, currentOp));

    if (newFlow == controlFlowEnd_)
        return false;

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());
    lastOp_ = currentOp;

    return true;

}

bool mv::OpModel::defaultStage_(Data::OpListIterator op)
{

    auto stageIt = addStage_();
    
    if (!addToStage_(stageIt, op))
        return false;

    return true;

}

mv::Data::OpListIterator mv::OpModel::checkInputTensor_(Data::TensorIterator inputTensor)
{

    if (inputTensor == tensorEnd_)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define source op - undefined input tensor" );
        return opEnd();
    }

    auto sourceIt = findSourceOp_(inputTensor);
    
    if (sourceIt == opEnd())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define source op - tensor '" + inputTensor->getName() + 
            "' does not belong to the computation model");
        return opEnd();
    }

    return sourceIt;

}

bool mv::OpModel::defineInputFlow_(Data::TensorIterator inputTensor, byte_type inputIdx, Data::OpListIterator op)
{

    if (op == opEnd())
        return false;

    auto inputSource = checkInputTensor_(inputTensor);
    if (inputSource == opEnd())
        return false;

    if (!op->setInput(inputTensor, inputIdx))
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define set input for op " + op->getName());
        return false;
    }

    Data::FlowListIterator inputFlow = dataGraph_.edge_insert(inputSource, op, allocator_.make_owner<DataFlow>(inputSource, op, inputTensor));
    
    if (inputFlow != dataFlowEnd_)
    {
        logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());
        return true;
    }
    else
        logger_.log(Logger::MessageType::MessageError, "Unable to define new flow for op " + op->getName());

    return false;

}

mv::Data::TensorIterator mv::OpModel::defineOp_(computation_graph::first_graph::node_list_iterator& opNode, Data::TensorIterator* inputs, byte_type numInputs)
{
    if (opNode == dataGraph_.node_end())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to allocate a new op");
        return tensorEnd_;
    }

    for (byte_type i = 0; i < numInputs; ++i)
    {

        if (!defineInputFlow_(inputs[i], i, opNode))
        {
            dataGraph_.node_erase(opNode);
            logger_.log(Logger::MessageType::MessageError, "Allocation of op failed due to input flow definition failure");
            return tensorEnd_;
        }

    }

    auto outputTensor = defineOutputTensor_(opNode, 0);
    (*opNode)->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*opNode)->toString());

    defaultControlFlow_(opNode);
    defaultStage_(opNode);

    return outputTensor;

}

mv::Data::OpListIterator mv::OpModel::switchContext(Control::OpListIterator& other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::Data::TensorIterator mv::OpModel::input(const Shape& shape, DType dType, Order order, const string& name)
{

    if (input_ != opEnd())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define input - already defined (multi-input models currently not supported");
        return tensorEnd_;
    }

    input_ = dataGraph_.node_insert(allocator_.make_owner<Op::Input>(shape, dType, order, name));
    
    if (input_ == opEnd())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to allocate a new input op");
        return tensorEnd_;
    }

    auto outputTensor = defineOutputTensor_(input_, 0);
    input_->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());

    lastOp_ = opsGraph_->get_second_iterator(input_);
    return outputTensor;

}

mv::Data::TensorIterator mv::OpModel::output(Data::TensorIterator inputTensor, const string& name)
{
    
    auto output = dataGraph_.node_insert(allocator_.make_owner<Op::Output>(name));

    if (output == dataGraph_.node_end())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to allocate a new output op");
        return tensorEnd_;
    }

    if (!defineInputFlow_(inputTensor, 0, output))
    {
        dataGraph_.node_erase(output);
        logger_.log(Logger::MessageType::MessageError, "Allocation of op output failed due to input flow definition failure");
        return tensorEnd_;
    }

    output_ = output;
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());
    defaultControlFlow_(output_);

    return inputTensor;

}

mv::Data::TensorIterator mv::OpModel::constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name)
{
    Data::OpListIterator constantIt = dataGraph_.node_insert(allocator_.make_owner<Op::Constant>(data, shape, dType, order, name));
    auto outputTensor = defineOutputTensor_(constantIt, 0);
    constantIt->setOutput(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + constantIt->toString());
    return outputTensor;
}

mv::Data::TensorIterator mv::OpModel::constant(float_type *data, size_type size, const Shape& shape, DType dType, Order order, const string& name)
{
    dynamic_vector<float_type> dataVec(data, size);
    return constant(dataVec, shape, dType, order, name);
}

mv::Data::TensorIterator mv::OpModel::conv2D(Data::TensorIterator inputTensor, Data::TensorIterator filtersTensor, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{

    auto conv = dataGraph_.node_insert(allocator_.make_owner<Op::Conv2D>(stride, padding, name));
    Data::TensorIterator inputs[] = {inputTensor, filtersTensor};
    return defineOp_(conv, inputs, 2);;

}

mv::Data::TensorIterator mv::OpModel::fullyConnected(Data::TensorIterator inputTensor, Data::TensorIterator weightsTensor, const string& name)
{

    auto fullyConnected = dataGraph_.node_insert(allocator_.make_owner<Op::FullyConnected>(name));
    Data::TensorIterator inputs[] = {inputTensor, weightsTensor};
    return defineOp_(fullyConnected, inputs, 2);
    
}

mv::Data::TensorIterator mv::OpModel::maxpool2D(Data::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{

    auto pool = dataGraph_.node_insert(allocator_.make_owner<Op::MaxPool2D>(kernelSize, stride, padding, name));
    Data::TensorIterator inputs[] = {inputTensor};
    return defineOp_(pool, inputs, 1);

}

mv::Data::TensorIterator mv::OpModel::avgpool2D(Data::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{

    auto pool = dataGraph_.node_insert(allocator_.make_owner<Op::AvgPool2D>(kernelSize, stride, padding, name));
    Data::TensorIterator inputs[] = {inputTensor};
    return defineOp_(pool, inputs, 1);

}

mv::Data::TensorIterator mv::OpModel::concat(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{

    Data::OpListIterator concat = dataGraph_.node_insert(allocator_.make_owner<Op::Concat>(name));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    return defineOp_(concat, inputs, 2);

}

mv::Data::TensorIterator mv::OpModel::batchNorm(Data::TensorIterator inputTensor, Data::TensorIterator meanTensor, Data::TensorIterator varianceTensor, Data::TensorIterator offsetTensor, Data::TensorIterator scaleTensor, float_type varianceEps, const string& name)
{

    Data::OpListIterator batchNorm = dataGraph_.node_insert(allocator_.make_owner<Op::BatchNorm>(varianceEps, name));
    Data::TensorIterator inputs[] = {inputTensor, meanTensor, varianceTensor, offsetTensor, scaleTensor};
    return defineOp_(batchNorm, inputs, 5);

}

mv::Data::TensorIterator mv::OpModel::scale(Data::TensorIterator inputTensor, Data::TensorIterator scaleTensor, const string& name)
{

    Data::OpListIterator scale = dataGraph_.node_insert(allocator_.make_owner<Op::Scale>(name));
    Data::TensorIterator inputs[] = {inputTensor, scaleTensor};
    return defineOp_(scale, inputs, 2);

}

mv::Data::TensorIterator mv::OpModel::relu(Data::TensorIterator inputTensor, const string& name)
{

    Data::OpListIterator relu = dataGraph_.node_insert(allocator_.make_owner<Op::ReLu>(name));
    Data::TensorIterator inputs[] = {inputTensor};
    return defineOp_(relu, inputs, 1);

}

mv::Data::TensorIterator mv::OpModel::softmax(Data::TensorIterator inputTensor, const string& name)
{

    Data::OpListIterator softmax = dataGraph_.node_insert(allocator_.make_owner<Op::Softmax>(name));
    Data::TensorIterator inputs[] = {inputTensor};
    return defineOp_(softmax, inputs, 1);

}

mv::Data::TensorIterator mv::OpModel::add(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{

    Data::OpListIterator add = dataGraph_.node_insert(allocator_.make_owner<Op::Add>(name));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    return defineOp_(add, inputs, 2);

}

mv::Data::TensorIterator mv::OpModel::subtract(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{

    Data::OpListIterator add = dataGraph_.node_insert(allocator_.make_owner<Op::Subtract>(name));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    return defineOp_(add, inputs, 2);

}

mv::Data::TensorIterator mv::OpModel::multiply(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{

    Data::OpListIterator add = dataGraph_.node_insert(allocator_.make_owner<Op::Multiply>(name));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    return defineOp_(add, inputs, 2);

}

mv::Data::TensorIterator mv::OpModel::divide(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{

    Data::OpListIterator add = dataGraph_.node_insert(allocator_.make_owner<Op::Divide>(name));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    return defineOp_(add, inputs, 2);

}

mv::Data::TensorIterator mv::OpModel::reshape(Data::TensorIterator inputTensor, const Shape& shape, const string& name)
{

    Data::OpListIterator reshape = dataGraph_.node_insert(allocator_.make_owner<Op::Reshape>(shape, name));
    Data::TensorIterator inputs[] = {inputTensor};
    return defineOp_(reshape, inputs, 1);

}

mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)
{
    return findSourceOp_(tensor);
}

bool mv::OpModel::addAttr(Data::OpListIterator opIt, const string& name, const Attribute& attr)
{

    return opIt->addAttr(name, attr);

}

bool mv::OpModel::isValid() const
{
    return ComputationModel::isValid();
}

mv::Data::OpListIterator mv::OpModel::getInput()
{
    return input_;
}

mv::Data::OpListIterator mv::OpModel::getOutput()
{
    return output_;
}

mv::Data::OpListIterator mv::OpModel::opEnd()
{
    return dataOpEnd_;
}


mv::GroupContext::MemberIterator mv::OpModel::addGroupElement(Data::OpListIterator newElement, GroupContext::GroupIterator group)
{

    allocator::owner_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);

}

bool mv::OpModel::removeGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group)
{
    allocator::owner_ptr<ComputationOp> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::dynamic_vector<mv::Shape> mv::OpModel::getInputShapes(Data::OpListIterator& op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostInput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

mv::dynamic_vector<mv::Shape> mv::OpModel::getOutputShapes(Data::OpListIterator& op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}