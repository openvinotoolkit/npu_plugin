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

mv::DataContext::OpListIterator mv::OpModel::output(DataContext::OpListIterator &predIt, const string &name)
{

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>(predIt->getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    output_ = dataGraph_.node_insert(allocator_.make_owner<Output>(logger_, **inTensorRes.first, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(predIt, output_, allocator_.make_owner<DataFlow>(logger_, predIt, output_, inTensorRes.first));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(output_);

    return output_;

}

mv::DataContext::OpListIterator mv::OpModel::conv(DataContext::OpListIterator &predIt, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>(predIt->getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    DataContext::OpListIterator convIt = dataGraph_.node_insert(allocator_.make_owner<Conv>(logger_, **inTensorRes.first, weights, strideX, strideY, padX, padY, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + convIt->toString());

    auto resultT = parameterTensors_->insert(allocator_.make_owner<PopulatedTensor>(logger_, convIt->getName() + "_" + "weights", convIt->getAttr("weights").getContent<ConstantTensor>()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*resultT.first)->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(predIt, convIt, allocator_.make_owner<DataFlow>(logger_, predIt, convIt, inTensorRes.first));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(convIt);
    defaultStage_(convIt);

    return convIt;
}

mv::DataContext::OpListIterator mv::OpModel::maxpool(DataContext::OpListIterator &predIt, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*predIt).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    DataContext::OpListIterator poolIt = dataGraph_.node_insert(allocator_.make_owner<MaxPool>(logger_, **inTensorRes.first, kernelShape, strideX, strideY, padX, padY, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + poolIt->toString());

    DataContext::FlowListIterator newFlow = dataGraph_.edge_insert(predIt, poolIt, allocator_.make_owner<DataFlow>(logger_, predIt, poolIt, inTensorRes.first));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + newFlow->toString());

    defaultControlFlow_(poolIt);
    defaultStage_(poolIt);

    return poolIt;
}

mv::DataContext::OpListIterator mv::OpModel::concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name)
{

    auto in0TensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>(input0->getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*in0TensorRes.first)->toString());

    auto in1TensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>(input1->getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*in1TensorRes.first)->toString());

    DataContext::OpListIterator concatIt = dataGraph_.node_insert(allocator_.make_owner<Concat>(logger_, **in0TensorRes.first, **in1TensorRes.first, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + concatIt->toString());

    DataContext::FlowListIterator input0Flow = dataGraph_.edge_insert(input0, concatIt, allocator_.make_owner<DataFlow>(logger_, input0, concatIt, in0TensorRes.first));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input0Flow->toString());

    DataContext::FlowListIterator input1Flow = dataGraph_.edge_insert(input1, concatIt, allocator_.make_owner<DataFlow>(logger_, input1, concatIt, in1TensorRes.first));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input1Flow->toString());

    defaultControlFlow_(concatIt);
    defaultStage_(concatIt);

    return concatIt;
}

bool mv::OpModel::addAttr(DataContext::OpListIterator &opIt, const string &name, const Attribute &attr)
{

    if  (attr.getType() == AttrType::TensorType)
    {
        auto resultT = parameterTensors_->insert(allocator_.make_owner<PopulatedTensor>(logger_, opIt->getName() + "_" + name, attr.getContent<ConstantTensor>()));
        if (!resultT.second)
            return false;

        bool resultA =  opIt->addAttr(name, attr);
        if (!resultA)
        {
            parameterTensors_->erase(*resultT.first);
            return false;
        }

        logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*resultT.first)->toString());

    }
    else
        return opIt->addAttr(name, attr);

    return false;

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