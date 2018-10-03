#include "include/mcm/computation/model/op_model.hpp"

mv::OpModel::OpModel(const std::string& name) :
ComputationModel(name)
{

}

/*mv::OpModel::OpModel(mv::json::Value& value) :
ComputationModel(value)
{

}*/

mv::OpModel::OpModel(ComputationModel& other) :
ComputationModel(other)
{

}

mv::OpModel::OpModel(CompositionalModel& model) :
ComputationModel(static_cast<OpModel&>(model))
{

}

mv::Data::OpListIterator mv::OpModel::checkInputTensor_(Data::TensorIterator inputTensor)
{

    if (inputTensor == tensorEnd())
    {
        log(Logger::MessageType::MessageError, "Unable to define source op - undefined input tensor" );
        return opEnd();
    }

    auto sourceIt = findSourceOp_(inputTensor);

    if (sourceIt == opEnd())
    {
        log(Logger::MessageType::MessageError, "Unable to define source op - tensor '" + inputTensor->getName() + 
            "' does not belong to the computation model");
        return opEnd();
    }

    return sourceIt;

}

mv::Data::TensorIterator mv::OpModel::defineOp_(computation_graph::first_graph::node_list_iterator& opNode, Data::TensorIterator* inputs, std::size_t numInputs)
{

    if (opNode == dataGraph_.node_end())
    {
        log(Logger::MessageType::MessageError, "Unable to allocate a new op");
        return tensorEnd();
    }

    for (std::size_t i = 0; i < numInputs; ++i)
    {

        if (defineFlow(inputs[i], opNode, i) == flowEnd())
        {
            dataGraph_.node_erase(opNode);
            log(Logger::MessageType::MessageError, "Allocation of op failed due to input flow definition failure");
            return tensorEnd();
        }

    }

    auto outputTensor = defineOutputTensor_(opNode, 0);
    (*opNode)->setOutputTensor(outputTensor, 0);
    log(Logger::MessageType::MessageInfo, "Defined " + (*opNode)->toString());

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

std::string mv::OpModel::getOpName_(OpType opType)
{
    return opType.toString() + "_" + std::to_string(opsCount(opType));
}

mv::Data::OpListIterator mv::OpModel::switchContext(Control::OpListIterator other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::Data::TensorIterator mv::OpModel::input(const Shape& shape, DType dType, Order order, const std::string& name)
{

    if (*input_ != opEnd())
    {
        log(Logger::MessageType::MessageError, "Unable to define input - already defined (multi-input models currently not supported");
        return tensorEnd();
    }

    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Input);

    *input_ = dataGraph_.node_insert(std::make_shared<op::Input>(shape, dType, order, opName));
    
    if (*input_ == opEnd())
    {
        log(Logger::MessageType::MessageError, "Unable to allocate a new input op");
        return tensorEnd();
    }

    auto outputTensor = defineOutputTensor_(*input_, 0);
    (*input_)->setOutputTensor(outputTensor, 0);
    incrementOpsCounter_(OpType::Input);
    log(Logger::MessageType::MessageInfo, "Defined " + (*input_)->toString());
    return outputTensor;

}

mv::Data::TensorIterator mv::OpModel::output(Data::TensorIterator inputTensor, const std::string& name)
{
    
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Output);
    
    auto outputIt = dataGraph_.node_insert(std::make_shared<op::Output>(opName));

    if (outputIt == dataGraph_.node_end())
    {
        log(Logger::MessageType::MessageError, "Unable to allocate a new output op");
        return tensorEnd();
    }

    if (defineFlow(inputTensor, outputIt, 0) == flowEnd())
    {
        dataGraph_.node_erase(outputIt);
        log(Logger::MessageType::MessageError, "Allocation of op output failed due to input flow definition failure");
        return tensorEnd();
    }

    incrementOpsCounter_(OpType::Output);
    *output_ = outputIt;
    log(Logger::MessageType::MessageInfo, "Defined " + (*output_)->toString());

    return inputTensor;

}

mv::Data::TensorIterator mv::OpModel::constant(const std::vector<double>& data, const Shape& shape, 
    DType dType, Order order, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Constant);
    Data::OpListIterator constantIt = dataGraph_.node_insert(std::make_shared<op::Constant>(data, shape, dType, order, opName));
    auto outputTensor = defineOutputTensor_(constantIt, 0);
    constantIt->setOutputTensor(outputTensor, 0);
    incrementOpsCounter_(OpType::Constant);
    log(Logger::MessageType::MessageInfo, "Defined " + constantIt->toString());
    return outputTensor;
}

mv::Data::TensorIterator mv::OpModel::conv2D(Data::TensorIterator inputTensor, Data::TensorIterator filtersTensor, 
    std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Conv2D);
    auto conv = dataGraph_.node_insert(std::make_shared<op::Conv2D>(stride, padding, opName));
    Data::TensorIterator inputs[] = {inputTensor, filtersTensor};
    auto result = defineOp_(conv, inputs, 2);;
    if (isValid(result))
        incrementOpsCounter_(OpType::Conv2D);
    return result;
}

mv::Data::TensorIterator mv::OpModel::matMul(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::MatMul);
    auto matMulIt = dataGraph_.node_insert(std::make_shared<op::MatMul>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(matMulIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::MatMul);
    return result;
}

mv::Data::TensorIterator mv::OpModel::maxpool2D(Data::TensorIterator inputTensor, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::MaxPool2D);
    auto poolIt = dataGraph_.node_insert(std::make_shared<op::MaxPool2D>(kernelSize, stride, padding, opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(poolIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::MaxPool2D);
    return result;
}

mv::Data::TensorIterator mv::OpModel::avgpool2D(Data::TensorIterator inputTensor, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::AvgPool2D);
    auto poolIt = dataGraph_.node_insert(std::make_shared<op::AvgPool2D>(kernelSize, stride, padding, opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(poolIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::AvgPool2D);
    return result;
}

mv::Data::TensorIterator mv::OpModel::concat(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Concat);
    Data::OpListIterator concatIt = dataGraph_.node_insert(std::make_shared<op::Concat>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(concatIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Concat);
    return result;
}

mv::Data::TensorIterator mv::OpModel::batchNorm(Data::TensorIterator inputTensor, Data::TensorIterator meanTensor, Data::TensorIterator varianceTensor, Data::TensorIterator offsetTensor, Data::TensorIterator scaleTensor, double varianceEps, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::BatchNorm);
    Data::OpListIterator batchNormIt = dataGraph_.node_insert(std::make_shared<op::BatchNorm>(varianceEps, opName));
    Data::TensorIterator inputs[] = {inputTensor, meanTensor, varianceTensor, offsetTensor, scaleTensor};
    auto result = defineOp_(batchNormIt, inputs, 5);
    if (isValid(result))
        incrementOpsCounter_(OpType::BatchNorm);
    return result;
}

mv::Data::TensorIterator mv::OpModel::scale(Data::TensorIterator inputTensor, Data::TensorIterator scaleTensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Scale);
    Data::OpListIterator scaleIt = dataGraph_.node_insert(std::make_shared<op::Scale>(opName));
    Data::TensorIterator inputs[] = {inputTensor, scaleTensor};
    auto result = defineOp_(scaleIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Scale);
    return result;
}

mv::Data::TensorIterator mv::OpModel::relu(Data::TensorIterator inputTensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::ReLU);
    Data::OpListIterator reluIt = dataGraph_.node_insert(std::make_shared<op::ReLU>(opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(reluIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::ReLU);
    return result;
}


mv::Data::TensorIterator mv::OpModel::prelu(Data::TensorIterator inputTensor, Data::TensorIterator negativeSlope, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::PReLU);

    std::cout << "Create Preluit" << std::endl;
    Data::OpListIterator preluIt = dataGraph_.node_insert(std::make_shared<op::PReLU>(opName));
    std::cout << "Create inputs" << std::endl;
    Data::TensorIterator inputs[] = {inputTensor, negativeSlope};

    std::cout << "DEFINE" << std::endl;

    auto result = defineOp_(preluIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::PReLU);
    return result;
}

mv::Data::TensorIterator mv::OpModel::conversion(Data::TensorIterator inputTensor, Order targetOrder, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Conversion);
    Data::OpListIterator conversionIt = dataGraph_.node_insert(std::make_shared<op::Conversion>(opName, targetOrder));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(conversionIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::Conversion);
    return result;
}
mv::Data::TensorIterator mv::OpModel::softmax(Data::TensorIterator inputTensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Softmax);
    Data::OpListIterator softmaxIt = dataGraph_.node_insert(std::make_shared<op::Softmax>(opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(softmaxIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::Softmax);
    return result;
}

mv::Data::TensorIterator mv::OpModel::add(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Add);
    Data::OpListIterator addIt = dataGraph_.node_insert(std::make_shared<op::Add>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(addIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Add);
    return result;
}

mv::Data::TensorIterator mv::OpModel::subtract(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Subtract);
    Data::OpListIterator subtractIt = dataGraph_.node_insert(std::make_shared<op::Subtract>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(subtractIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Subtract);
    return result;
}

mv::Data::TensorIterator mv::OpModel::multiply(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Multiply);
    Data::OpListIterator multiplyIt = dataGraph_.node_insert(std::make_shared<op::Multiply>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(multiplyIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Multiply);
    return result;

}

mv::Data::TensorIterator mv::OpModel::divide(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
{   
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Divide);
    Data::OpListIterator divideIt = dataGraph_.node_insert(std::make_shared<op::Divide>(opName));
    Data::TensorIterator inputs[] = {input0Tensor, input1Tensor};
    auto result = defineOp_(divideIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Divide);
    return result;
}

mv::Data::TensorIterator mv::OpModel::reshape(Data::TensorIterator inputTensor, const Shape& shape, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Reshape);
    Data::OpListIterator reshapeIt = dataGraph_.node_insert(std::make_shared<op::Reshape>(shape, opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(reshapeIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::Reshape);
    return result;
}

mv::Data::TensorIterator mv::OpModel::bias(Data::TensorIterator inputTensor, Data::TensorIterator biasesTensor, const std::string& name)
{   
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Bias);
    Data::OpListIterator biasIt = dataGraph_.node_insert(std::make_shared<op::Bias>(opName));
    Data::TensorIterator inputs[] = {inputTensor, biasesTensor};
    auto result = defineOp_(biasIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::Bias);
    return result;
}

mv::Data::TensorIterator mv::OpModel::fullyConnected(Data::TensorIterator inputTensor, Data::TensorIterator weightsTensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::FullyConnected);

    Data::OpListIterator fullyConnectedIt = dataGraph_.node_insert(std::make_shared<op::FullyConnected>(opName));
    Data::TensorIterator inputs[] = {inputTensor, weightsTensor};
    auto result = defineOp_(fullyConnectedIt, inputs, 2);
    if (isValid(result))
        incrementOpsCounter_(OpType::FullyConnected);
    return result;

}

mv::Data::TensorIterator mv::OpModel::dropOut(Data::TensorIterator inputTensor, const std::string& name)
{
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::DropOut);

    Data::OpListIterator dropOutIt = dataGraph_.node_insert(std::make_shared<op::DropOut>(opName));
    Data::TensorIterator inputs[] = {inputTensor};
    auto result = defineOp_(dropOutIt, inputs, 1);
    if (isValid(result))
        incrementOpsCounter_(OpType::DropOut);
    return result;

}

mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)
{
    return findSourceOp_(tensor);
}

bool mv::OpModel::removeOp(Data::OpListIterator op)
{

    if (op == opEnd())
        return false;

    for (std::size_t j = 0; j < op->outputSlots(); ++j)
    {
        tensorsSources_->erase(op->getOutputTensor(j)->getName());
        flowTensors_->erase(op->getOutputTensor(j));
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

mv::Data::FlowListIterator mv::OpModel::defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx)
{

    if (!isValid(sourceTensor))
    {
        log(Logger::MessageType::MessageError, "Unable to define source op - undefined input tensor" );
        return flowEnd();
    }

    if (!isValid(sinkOp))
        return flowEnd();

    auto sourceOp = findSourceOp_(sourceTensor);

    if (!isValid(sourceOp))
    {
        log(Logger::MessageType::MessageError, "Unable to define source op - tensor '" + sourceTensor->getName() + 
            "' does not belong to the computation model");
        return flowEnd();
    }

    if (!sinkOp->setInputTensor(sourceTensor, inputIdx))
    {
        log(Logger::MessageType::MessageError, "Unable to define set input for op " + sinkOp->getName());
        return flowEnd();
    }

    Data::FlowListIterator inputFlow = dataGraph_.edge_insert(sourceOp, sinkOp, std::make_shared<DataFlow>(sourceOp, 0, sinkOp, inputIdx, sourceTensor));
    
    if (inputFlow != *dataFlowEnd_)
    {
        log(Logger::MessageType::MessageInfo, "Defined " + inputFlow->toString());
        return inputFlow;
    }
    else
    {
        log(Logger::MessageType::MessageError, "Unable to define new flow for op " + sourceOp->getName());
    }

    return flowEnd();

}

mv::Data::FlowListIterator mv::OpModel::defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx)
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

mv::Data::OpListIterator mv::OpModel::opBegin() const
{
    return dataGraph_.node_begin();
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

    std::shared_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);

}

bool mv::OpModel::removeGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group)
{
    std::shared_ptr<ComputationOp> ptr = element;
    return removeGroupElement_(ptr, group);
}

std::vector<mv::Shape> mv::OpModel::getInputShapes(Data::OpListIterator& op)
{

    std::vector<Shape> shapes;

    for (auto it = op.leftmostInput(); it != *dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

std::vector<mv::Shape> mv::OpModel::getOutputShapes(Data::OpListIterator& op)
{

    std::vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != *dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

std::size_t mv::OpModel::opsCount() const
{
    return dataGraph_.node_size();
}

std::size_t mv::OpModel::opsCount(OpType opType) const
{
    if (opsCounter_->find(opType) != opsCounter_->end())
        return opsCounter_->at(opType);
    return 0;
}

long long unsigned mv::OpModel::parametersCount() const
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

void mv::OpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    op->set(name, attr);
}

std::string mv::OpModel::getLogID() const
{
    return "OpModel";
}
