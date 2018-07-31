#include "include/mcm/computation/model/op_model.hpp"

mv::OpModel::OpModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::OpModel::OpModel(mv::json::Value& value, Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(value, verboseLevel, logTime)
{

}

mv::OpModel::OpModel(const ComputationModel& other) :
ComputationModel(other)
{

}

mv::OpModel::OpModel(const CompositionalModel& model) :
ComputationModel(static_cast<const OpModel&>(model))
{

}

mv::Data::OpListIterator mv::OpModel::checkInputTensor_(Data::TensorIterator inputTensor)
{

    if (inputTensor == tensorEnd())
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

mv::Data::TensorIterator mv::OpModel::defineOp_(computation_graph::first_graph::node_list_iterator& opNode, Data::TensorIterator* inputs, byte_type numInputs)
{

    if (opNode == dataGraph_.node_end())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to allocate a new op");
        return tensorEnd();
    }

    for (byte_type i = 0; i < numInputs; ++i)
    {

        if (defineFlow(inputs[i], opNode, i) == flowEnd())
        {
            dataGraph_.node_erase(opNode);
            logger_.log(Logger::MessageType::MessageError, "Allocation of op failed due to input flow definition failure");
            return tensorEnd();
        }

    }

    auto outputTensor = defineOutputTensor_(opNode, 0);
    (*opNode)->setOutputTensor(outputTensor, 0);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*opNode)->toString());

    return outputTensor;

}

void mv::OpModel::incrementOpsCounter_(OpType opType)
{
    if (opsCounter_->find(opType) == opsCounter_->end())
    {  
        opsCounter_->emplace(opType, 1);
    }
    else
        ++opsCounter_->at(opType);
}
void mv::OpModel::decrementOpsCounter_(OpType opType)
{
    if (opsCounter_->find(opType) == opsCounter_->end())
        return;
    else
    {
        if (opsCounter_->at(opType) > 0)
            --opsCounter_->at(opType);
    }
}

mv::string mv::OpModel::getOpName_(OpType opType)
{
    return Printable::toString(opType) + "_" + Printable::toString(opsCount(opType));
}

mv::Data::OpListIterator mv::OpModel::switchContext(Control::OpListIterator other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::Data::TensorIterator mv::OpModel::input(const Shape& shape, DType dType, Order order, const string& name)
{

    if (*input_ != opEnd())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define input - already defined (multi-input models currently not supported");
        return tensorEnd();
    }

    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Input);

    *input_ = dataGraph_.node_insert(allocator_.make_owner<op::Input>(shape, dType, order, opName));
    
    if (*input_ == opEnd())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to allocate a new input op");
        return tensorEnd();
    }

    auto outputTensor = defineOutputTensor_(*input_, 0);
    (*input_)->setOutputTensor(outputTensor, 0);
    incrementOpsCounter_(OpType::Input);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*input_)->toString());
    return outputTensor;

}

mv::Data::TensorIterator mv::OpModel::output(Data::TensorIterator inputTensor, const string& name)
{
    
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Output);
    
    auto outputIt = dataGraph_.node_insert(allocator_.make_owner<op::Output>(opName));

    if (outputIt == dataGraph_.node_end())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to allocate a new output op");
        return tensorEnd();
    }

    if (defineFlow(inputTensor, outputIt, 0) == flowEnd())
    {
        dataGraph_.node_erase(outputIt);
        logger_.log(Logger::MessageType::MessageError, "Allocation of op output failed due to input flow definition failure");
        return tensorEnd();
    }

    incrementOpsCounter_(OpType::Output);
    *output_ = outputIt;
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*output_)->toString());

    return inputTensor;

}

mv::Data::TensorIterator mv::OpModel::constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Constant);
    Data::OpListIterator constantIt = dataGraph_.node_insert(allocator_.make_owner<op::Constant>(data, shape, dType, order, opName));
    auto outputTensor = defineOutputTensor_(constantIt, 0);
    constantIt->setOutputTensor(outputTensor, 0);
    incrementOpsCounter_(OpType::Constant);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + constantIt->toString());
    return outputTensor;
}

mv::Data::TensorIterator mv::OpModel::conv2D(Data::TensorIterator inputTensor, Data::TensorIterator filtersTensor, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Conv2D);
    auto conv = dataGraph_.node_insert(allocator_.make_owner<op::Conv2D>(stride, padding, opName));
    Data::TensorIterator inputs[] = {inputTensor, filtersTensor};
    auto result = defineOp_(conv, inputs, 2);;
    if (isValid(result))
        incrementOpsCounter_(OpType::Conv2D);
    return result;
}

mv::Data::TensorIterator mv::OpModel::matMul(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::MatMul);
    auto matMulIt = dataGraph_.node_insert(allocator_.make_owner<op::MatMul>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(matMulIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::MatMul);
    return result;
}

mv::Data::TensorIterator mv::OpModel::maxpool2D(Data::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::MaxPool2D);
    auto poolIt = dataGraph_.node_insert(allocator_.make_owner<op::MaxPool2D>(kernelSize, stride, padding, opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(poolIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::MaxPool2D);
    return result;
}

mv::Data::TensorIterator mv::OpModel::avgpool2D(Data::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::AvgPool2D);
    auto poolIt = dataGraph_.node_insert(allocator_.make_owner<op::AvgPool2D>(kernelSize, stride, padding, opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(poolIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::AvgPool2D);
    return result;
}

mv::Data::TensorIterator mv::OpModel::concat(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Concat);
    Data::OpListIterator concatIt = dataGraph_.node_insert(allocator_.make_owner<op::Concat>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(concatIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Concat);
    return result;
}

mv::Data::TensorIterator mv::OpModel::batchNorm(Data::TensorIterator inputTensor, Data::TensorIterator meanTensor, Data::TensorIterator varianceTensor, Data::TensorIterator offsetTensor, Data::TensorIterator scaleTensor, float_type varianceEps, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::BatchNorm);
    Data::OpListIterator batchNormIt = dataGraph_.node_insert(allocator_.make_owner<op::BatchNorm>(varianceEps, opName));
    Data::TensorIterator inputs[] = {inputTensor, meanTensor, varianceTensor, offsetTensor, scaleTensor};
    auto result = defineOp_(batchNormIt, inputs, 5);
    if (isValid(result))
        incrementOpsCounter_(OpType::BatchNorm);
    return result;
}

mv::Data::TensorIterator mv::OpModel::scale(Data::TensorIterator inputTensor, Data::TensorIterator scaleTensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Scale);
    Data::OpListIterator scaleIt = dataGraph_.node_insert(allocator_.make_owner<op::Scale>(opName));
    Data::TensorIterator inputs[] = {inputTensor, scaleTensor};
    auto result = defineOp_(scaleIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Scale);
    return result;
}

mv::Data::TensorIterator mv::OpModel::relu(Data::TensorIterator inputTensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::ReLU);
    Data::OpListIterator reluIt = dataGraph_.node_insert(allocator_.make_owner<op::ReLU>(opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(reluIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::ReLU);
    return result;
}

mv::Data::TensorIterator mv::OpModel::softmax(Data::TensorIterator inputTensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Softmax);
    Data::OpListIterator softmaxIt = dataGraph_.node_insert(allocator_.make_owner<op::Softmax>(opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(softmaxIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::Softmax);
    return result;
}

mv::Data::TensorIterator mv::OpModel::add(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Add);
    Data::OpListIterator addIt = dataGraph_.node_insert(allocator_.make_owner<op::Add>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(addIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Add);
    return result;
}

mv::Data::TensorIterator mv::OpModel::subtract(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Subtract);
    Data::OpListIterator subtractIt = dataGraph_.node_insert(allocator_.make_owner<op::Subtract>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(subtractIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Subtract);
    return result;
}

mv::Data::TensorIterator mv::OpModel::multiply(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Multiply);
    Data::OpListIterator multiplyIt = dataGraph_.node_insert(allocator_.make_owner<op::Multiply>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(multiplyIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Multiply);
    return result;

}

mv::Data::TensorIterator mv::OpModel::divide(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{   
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Divide);
    Data::OpListIterator divideIt = dataGraph_.node_insert(allocator_.make_owner<op::Divide>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(divideIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Divide);
    return result;
}

mv::Data::TensorIterator mv::OpModel::reshape(Data::TensorIterator inputTensor, const Shape& shape, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Reshape);
    Data::OpListIterator reshapeIt = dataGraph_.node_insert(allocator_.make_owner<op::Reshape>(shape, opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(reshapeIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::Reshape);
    return result;
}

mv::Data::TensorIterator mv::OpModel::bias(Data::TensorIterator inputTensor, Data::TensorIterator biasesTensor, const string& name)
{   
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Bias);
    Data::OpListIterator biasIt = dataGraph_.node_insert(allocator_.make_owner<op::Bias>(opName));
    Data::TensorIterator inputs[] = {inputTensor, biasesTensor};
    auto result = defineOp_(biasIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Bias);
    return result;
}

mv::Data::TensorIterator mv::OpModel::fullyConnected(Data::TensorIterator inputTensor, Data::TensorIterator weightsTensor, const string& name)
{
    string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::FullyConnected);

    Data::OpListIterator fullyConnectedIt = dataGraph_.node_insert(allocator_.make_owner<op::FullyConnected>(opName));
    Data::TensorIterator inputs[] = {inputTensor, weightsTensor};
    auto result = defineOp_(fullyConnectedIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::FullyConnected);
    return result;
    
}

mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)
{
    return findSourceOp_(tensor);
}

bool mv::OpModel::addAttr(Data::OpListIterator opIt, const string& name, const Attribute& attr)
{

    return opIt->addAttr(name, attr);

}

bool mv::OpModel::removeOp(Data::OpListIterator op)
{

    if (op == opEnd())
        return false;

    for (byte_type j = 0; j < op->outputSlots(); ++j)
    {
        flowTensors_->erase(op->getOutputTensor(j));
        tensorsSources_->erase(op->getOutputTensor(j)->getName());
    }

    auto opCounterIt = opsCounter_->find(op->getOpType());
    if (opCounterIt != opsCounter_->end())
    {
        --opCounterIt->second;
        if (opCounterIt->second == 0)
            opsCounter_->erase(opCounterIt);
            
    }

    dataGraph_.node_erase(op);
    
    return true;

}

mv::Data::FlowListIterator mv::OpModel::defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, byte_type inputIdx)
{
    
    if (!isValid(sourceTensor))
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define source op - undefined input tensor" );
        return flowEnd();
    }

    if (!isValid(sinkOp))
        return flowEnd();

    auto sourceOp = findSourceOp_(sourceTensor);
    
    if (!isValid(sourceOp))
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define source op - tensor '" + sourceTensor->getName() + 
            "' does not belong to the computation model");
        return flowEnd();
    }

    if (!sinkOp->setInputTensor(sourceTensor, inputIdx))
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define set input for op " + sinkOp->getName());
        return flowEnd();
    }

    Data::FlowListIterator inputFlow = dataGraph_.edge_insert(sourceOp, sinkOp, allocator_.make_owner<DataFlow>(sourceOp, 0, sinkOp, inputIdx, sourceTensor));
    
    if (inputFlow != *dataFlowEnd_)
    {
        logger_.log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());
        return inputFlow;
    }
    else
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define new flow for op " + sourceOp->getName());
    }

    return flowEnd();

}

mv::Data::FlowListIterator mv::OpModel::defineFlow(Data::OpListIterator sourceOp, byte_type outputIdx, Data::OpListIterator sinkOp, byte_type inputIdx)
{

    auto sourceTensor = sourceOp->getOutputTensor(outputIdx);
    return defineFlow(sourceTensor, sinkOp, inputIdx);
    
}

bool mv::OpModel::undefineFlow(Data::FlowListIterator flow)
{

    if (!ComputationModel::isValid(flow))
        return false;

    dataGraph_.edge_erase(flow);
    return true;

}

bool mv::OpModel::isValid() const
{
    return ComputationModel::isValid();
}

bool mv::OpModel::isValid(const Data::TensorIterator &it) const
{
    return ComputationModel::isValid(it);
}

bool mv::OpModel::isValid(const Data::OpListIterator &it) const
{
    return ComputationModel::isValid(it);
}

mv::Data::OpListIterator mv::OpModel::getInput()
{
    return *input_;
}

mv::Data::OpListIterator mv::OpModel::getOutput()
{
    return *output_;
}

mv::Data::OpListIterator mv::OpModel::opEnd() const
{
    return *dataOpEnd_;
}

mv::Data::FlowListIterator mv::OpModel::flowEnd() const
{
    return *dataFlowEnd_;
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

    for (auto it = op.leftmostInput(); it != *dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

mv::dynamic_vector<mv::Shape> mv::OpModel::getOutputShapes(Data::OpListIterator& op)
{

    dynamic_vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != *dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

unsigned mv::OpModel::opsCount() const
{
    return dataGraph_.node_size();
}

unsigned mv::OpModel::opsCount(OpType opType) const
{
    if (opsCounter_->find(opType) != opsCounter_->end())
        return opsCounter_->at(opType);
    return 0;
}

unsigned mv::OpModel::parametersCount() const
{

    unsigned result = 0;

    for (auto it = *input_; it != opEnd(); ++it)
    {
        if (it->getOpType() == OpType::Constant)
        {
            result += it->getOutputTensor(0)->getShape().totalSize();
        }
    }

    return result;

}