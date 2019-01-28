#include "include/mcm/computation/model/base_op_model.hpp"

mv::BaseOpModel::BaseOpModel(const std::string& name) :
ComputationModel(name)
{
    log(Logger::MessageType::Debug, "Initialized");
}

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
            const std::vector<std::pair<std::string, Attribute>> & args, std::string name, bool checkInputSize, bool checkArgs)
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
    
    auto opNode = dataGraph_.node_insert(Op(*this, opType, name, inputs, args, checkInputSize, checkArgs));

    incrementOpsInstanceCounter_(opType);
    incrementOpsIndexCounter_(opType);

    ops_->emplace(name, opNode);

    for (std::size_t i = 0; i < (*opNode).inputSlots(); ++i)
        defineFlow(inputs[i], opNode, i);

    log(Logger::MessageType::Info, "Defined " + (*opNode).toString());

    // Assumes single input/output
    if (opType == "Input")
    {
        if (*input_ == opEnd())
            *input_ = opNode;
        else
            throw LogicError(*this, "Attempt of multi-input model definiton - currently unsupported");
    }
    else if (opType == "Output")
    {
        if (*output_ == opEnd())
            *output_ = opNode;
        else
            throw LogicError(*this, "Attempt of multi-output model definiton - currently unsupported");
    }
    
    if ((*opNode).outputSlots() > 0)
        return (*opNode).getOutputTensor(0);

    return tensorEnd();

}

void mv::BaseOpModel::removeOp(Data::OpListIterator op)
{

    if (op == opEnd())
        throw ArgumentError(*this, "op:iterator", "end", "Invalid iterator passed for op removal");

    //Removing input/output data flows from the model
    //There is no actual need to call undefineFlow, as the graph structure will be handled by dataGraph_.node_erase(op)
    //But undefineFlow also removes the flow information from the tensor, so it's better to use it

    for (Data::FlowSiblingIterator sourceFlow(op.leftmostInput()); sourceFlow != flowEnd(); ++sourceFlow)
        undefineFlow(sourceFlow);
        //dataFlows_->erase(sourceFlow->getName());

    for (Data::FlowSiblingIterator sinkFlow(op.leftmostOutput()); sinkFlow != flowEnd(); ++sinkFlow)
    {
        sinkFlow.sink()->set<bool>("invalid", true);
        undefineFlow(sinkFlow);
    }

    //Removing output tensors from the model
    for (std::size_t j = 0; j < op->outputSlots(); ++j)
        tensors_->erase(op->getOutputTensor(j)->getName());

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
    
    if(!sourceTensor->hasAttr("flows"))
    {
        std::set<std::string> toSet;
        sourceTensor->set<std::set<std::string>>("flows", toSet);
    }

    sourceTensor->get<std::set<std::string>>("flows").insert(inputFlow->getName());
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

    if(!flow->getTensor()->hasAttr("flows"))
        log(Logger::MessageType::Error, flow->getTensor()->getName() + " is in a flow but has no attribute flows");

    flow->getTensor()->get<std::set<std::string>>("flows").erase(flow->getName());
    dataFlows_->erase(flow->getName());
    dataGraph_.edge_erase(flow);

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

void mv::BaseOpModel::addGroupElement(Data::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including op to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including op to a group");

    group->include(element);
}

void mv::BaseOpModel::removeGroupElement(Data::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    group->exclude(element);
}

std::vector<mv::Shape> mv::BaseOpModel::getInputShapes(Data::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed getting inputs shapes");

    std::vector<Shape> shapes;
    for (auto it = op.leftmostInput(); it != *dataFlowEnd_; ++it)
        shapes.push_back(it->getTensor()->getShape());
    return shapes;

}

std::vector<mv::Shape> mv::BaseOpModel::getOutputShapes(Data::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed getting outputs shap");

    std::vector<Shape> shapes;
    for (auto it = op.leftmostOutput(); it != *dataFlowEnd_; ++it)
        shapes.push_back(it->getTensor()->getShape());

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
