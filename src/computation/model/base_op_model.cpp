#include "include/mcm/computation/model/base_op_model.hpp"

mv::BaseOpModel::BaseOpModel(const std::string& name) :
ComputationModel(name)
{
    log(Logger::MessageType::Debug, "Initialized");
}

/*mv::BaseOpModel::BaseOpModel(mv::json::Value& value) :
ComputationModel(value)
{

}*/

mv::BaseOpModel::BaseOpModel(ComputationModel& other) :
ComputationModel(other)
{
    log(Logger::MessageType::Debug, "Bound");
}

mv::BaseOpModel::~BaseOpModel()
{
    log(Logger::MessageType::Debug, "Deleted");
}

mv::Data::OpListIterator mv::BaseOpModel::switchContext(Control::OpListIterator other)
{
    return opsGraph_->get_first_iterator(other);
}

/*mv::Data::TensorIterator mv::BaseOpModel::input(const Shape& shape, DType dType, Order order, const std::string& name)
{

    if (*input_ != opEnd())
    {
        log(Logger::MessageType::Error, "Unable to define input - already defined (multi-input models currently not supported");
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
        log(Logger::MessageType::Error, "Unable to allocate a new input op");
        return tensorEnd();
    }

    auto outputTensor = defineOutputTensor_(*input_, 0);
    (*input_)->setOutputTensor(outputTensor, 0);
    incrementOpsCounter_(OpType::Input);
    log(Logger::MessageType::Info, "Defined " + (*input_)->toString());
    return outputTensor;

}

mv::Data::TensorIterator mv::BaseOpModel::output(Data::TensorIterator inputTensor, const std::string& name)
{
    
    std::string opName;
    if (name != "")
        opName = name;
    else
        opName = getOpName_(OpType::Output);
    
    auto outputIt = dataGraph_.node_insert(std::make_shared<op::Output>(opName));

    if (outputIt == dataGraph_.node_end())
    {
        log(Logger::MessageType::Error, "Unable to allocate a new output op");
        return tensorEnd();
    }

    defineFlow(inputTensor, outputIt, 0);
    incrementOpsCounter_(OpType::Output);
    *output_ = outputIt;
    log(Logger::MessageType::Info, "Defined " + (*output_)->toString());

    return inputTensor;

}

mv::Data::TensorIterator mv::BaseOpModel::constant(const std::vector<double>& data, const Shape& shape, 
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
    log(Logger::MessageType::Info, "Defined " + constantIt->toString());
    return outputTensor;
}

mv::Data::TensorIterator mv::BaseOpModel::conv2D(Data::TensorIterator inputTensor, Data::TensorIterator filtersTensor, 
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

mv::Data::TensorIterator mv::BaseOpModel::matMul(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::maxpool2D(Data::TensorIterator inputTensor, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::avgpool2D(Data::TensorIterator inputTensor, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::concat(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::batchNorm(Data::TensorIterator inputTensor, Data::TensorIterator meanTensor, Data::TensorIterator varianceTensor, Data::TensorIterator offsetTensor, Data::TensorIterator scaleTensor, double varianceEps, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::scale(Data::TensorIterator inputTensor, Data::TensorIterator scaleTensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::relu(Data::TensorIterator inputTensor, const std::string& name)
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


mv::Data::TensorIterator mv::BaseOpModel::prelu(Data::TensorIterator inputTensor, Data::TensorIterator negativeSlope, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::conversion(Data::TensorIterator inputTensor, Order targetOrder, const std::string& name)
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
mv::Data::TensorIterator mv::BaseOpModel::softmax(Data::TensorIterator inputTensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::add(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::subtract(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::multiply(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::divide(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::reshape(Data::TensorIterator inputTensor, const Shape& shape, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::bias(Data::TensorIterator inputTensor, Data::TensorIterator biasesTensor, const std::string& name)
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

mv::Data::TensorIterator mv::BaseOpModel::fullyConnected(Data::TensorIterator inputTensor, Data::TensorIterator weightsTensor, const std::string& name)
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

}*/

mv::Data::OpListIterator mv::BaseOpModel::getSourceOp(Data::TensorIterator tensor)
{

    if (!tensor->hasAttr("sourceOp"))
        return opEnd();
    
    auto it = ops_->find(tensor->get<std::string>("sourceOp"));
    if (it == ops_->end())
        throw RuntimeError(*this, "Source op " + tensor->get<std::string>("sourceOp") + " of tensor " +
            tensor->getName() + " does not belong to the model");
    
    return it->second;

}

mv::Data::TensorIterator mv::BaseOpModel::defineOp(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
            std::initializer_list<std::pair<std::string, Attribute>> args, std::string name)
{

    if (name.empty())
    {
        if (opsIndexCounter_->find(opType) != opsIndexCounter_->end())
            name = opType + "_" + std::to_string(opsIndexCounter_->at(opType));
        else
            name = opType + "_0";
    }

    if (ops_->find(name) != ops_->end())
        throw ArgumentError(*this, "op:name", name, "Duplicated op name");
    
    auto opNode = dataGraph_.node_insert(Op(*this, opType, name, inputs, args));

    for (std::size_t i = 0; i < (*opNode).inputSlots(); ++i)
        defineFlow(inputs[i], opNode, i);

    incrementOpsInstanceCounter_(opType);
    incrementOpsIndexCounter_(opType);

    ops_->emplace(name, opNode);

    log(Logger::MessageType::Info, "Defined " + (*opNode).toString());

    if ((*opNode).outputSlots() > 0)
        return (*opNode).getOutputTensor(0);

    return tensorEnd();

}

void mv::BaseOpModel::removeOp(Data::OpListIterator op)
{

    if (op == opEnd())
        throw ArgumentError(*this, "op:iterator", "end", "Invalid iterator passed for op removal");

    for (std::size_t j = 0; j < op->outputSlots(); ++j)
        tensors_->erase(op->getOutputTensor(j));

    decrementOpsInstanceCounter_(op->getOpType());
    ops_->erase(op->getName());

    log(Logger::MessageType::Info, "Removed " + op->toString());

    dataGraph_.node_erase(op);
    
}

mv::Data::FlowListIterator mv::BaseOpModel::defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx)
{

    if (!isValid(sourceTensor))
        throw ArgumentError(*this, "sourceTensor", "invalid", "Invalid tensor passed for the data flow definition");

    if (!isValid(sinkOp))
        throw ArgumentError(*this, "sinkOp", "invalid", "Invalid sink op passed for the data flow definition");

    auto sourceOp = getSourceOp(sourceTensor);
    if (sourceOp == opEnd())
        throw ArgumentError(*this, "sourceTensor", "sourceless", "Defining flow using a tensor that does not have a source op is illegal");

    Data::FlowListIterator inputFlow = dataGraph_.edge_insert(sourceOp, sinkOp, DataFlow(*this, sourceOp, 0, sinkOp, inputIdx, sourceTensor));
    
    dataFlows_->emplace(inputFlow->getName(), inputFlow);
    log(Logger::MessageType::Info, "Defined " + inputFlow->toString());
    return inputFlow;

}

mv::Data::FlowListIterator mv::BaseOpModel::defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx)
{

    auto sourceTensor = sourceOp->getOutputTensor(outputIdx);
    return defineFlow(sourceTensor, sinkOp, inputIdx);

}

void mv::BaseOpModel::undefineFlow(Data::FlowListIterator flow)
{

    if (!ComputationModel::isValid(flow))
        throw ArgumentError(*this, "flow:iterator", "invalid", "Invalid flow passed for deletion");

    log(Logger::MessageType::Info, "Removed " + flow->toString());
    dataFlows_->erase(flow->getName());
    dataGraph_.edge_erase(flow);

}

bool mv::BaseOpModel::isValid() const
{
    return ComputationModel::isValid();
}

bool mv::BaseOpModel::isValid(const Data::TensorIterator &it) const
{
    return ComputationModel::isValid(it);
}

bool mv::BaseOpModel::isValid(const Data::OpListIterator &it) const
{
    return ComputationModel::isValid(it);
}

mv::Data::OpListIterator mv::BaseOpModel::getInput()
{
    return *input_;
}

mv::Data::OpListIterator mv::BaseOpModel::getOutput()
{
    return *output_;
}

mv::Data::OpListIterator mv::BaseOpModel::opBegin() const
{
    return dataGraph_.node_begin();
}

mv::Data::OpListIterator mv::BaseOpModel::opEnd() const
{
    return *dataOpEnd_;
}

mv::Data::FlowListIterator mv::BaseOpModel::flowEnd() const
{
    return *dataFlowEnd_;
}

/*mv::GroupContext::MemberIterator mv::BaseOpModel::addGroupElement(Data::OpListIterator newElement, GroupContext::GroupIterator group)
{

    std::shared_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);

}

bool mv::BaseOpModel::removeGroupElement(Data::OpListIterator element, GroupContext::GroupIterator group)
{
    std::shared_ptr<ComputationOp> ptr = element;
    return removeGroupElement_(ptr, group);
}*/

std::vector<mv::Shape> mv::BaseOpModel::getInputShapes(Data::OpListIterator& op)
{

    std::vector<Shape> shapes;

    for (auto it = op.leftmostInput(); it != *dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

std::vector<mv::Shape> mv::BaseOpModel::getOutputShapes(Data::OpListIterator& op)
{

    std::vector<Shape> shapes;

    for (auto it = op.leftmostOutput(); it != *dataFlowEnd_; ++it)
    {
        shapes.push_back(it->getTensor()->getShape());
    }

    return shapes;

}

std::size_t mv::BaseOpModel::opsCount() const
{
    return dataGraph_.node_size();
}

std::size_t mv::BaseOpModel::opsCount(const std::string& opType) const
{
    if (opsInstanceCounter_->find(opType) != opsInstanceCounter_->end())
        return opsInstanceCounter_->at(opType);
    return 0;
}

long long unsigned mv::BaseOpModel::parametersCount() const
{

    unsigned result = 0;

    for (auto it = *input_; it != opEnd(); ++it)
    {
        if (it->getOpType() == "Constant")
        {
            result += it->getOutputTensor(0)->getShape().totalSize();
        }
    }

    return result;

}

void mv::BaseOpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    op->set(name, attr);
}

std::string mv::BaseOpModel::getLogID() const
{
    return "OpModel:" + name_;
}