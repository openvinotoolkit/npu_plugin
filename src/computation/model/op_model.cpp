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

mv::DataContext::OpListIterator mv::OpModel::input(const Shape &shape, DType dType, Order order, const string &name)
{

    input_ = dataGraph_.node_insert(allocator_.make_owner<Input>(logger_, shape, dType, order, name));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*input_)->toString());

    lastOp_ = opsGraph_->get_second_iterator(input_);

    return input_;

}

mv::DataContext::OpListIterator mv::OpModel::output(DataContext::OpListIterator &predIt, const string &name)
{

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*predIt).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    output_ = dataGraph_.node_insert(predIt, allocator_.make_owner<Output>(logger_, **inTensorRes.first, name), *inTensorRes.first);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*output_)->toString());

    auto currentOp = opsGraph_->get_second_iterator(output_);
    controlGraph_.edge_insert(lastOp_, currentOp, ControlFlow());
    lastOp_ = currentOp;

    return output_;

}

mv::DataContext::OpListIterator mv::OpModel::conv(DataContext::OpListIterator &predIt, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{


    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*predIt).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    computation_graph::first_graph::node_list_iterator convIt = dataGraph_.node_insert(predIt, allocator_.make_owner<Conv>(logger_, **inTensorRes.first, weights, strideX, strideY, padX, padY, name), *inTensorRes.first);
    auto resultT = parameterTensors_->insert(allocator_.make_owner<PopulatedTensor>(logger_, (*convIt)->getName() + "_" + "weights", (*convIt)->getAttr("weights").getContent<ConstantTensor>()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*resultT.first)->toString());
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*convIt)->toString());

    auto currentOp = opsGraph_->get_second_iterator(convIt);
    controlGraph_.edge_insert(lastOp_, currentOp, ControlFlow());
    lastOp_ = currentOp;

    return convIt;
}

mv::DataContext::OpListIterator mv::OpModel::maxpool(DataContext::OpListIterator &predIt, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*predIt).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    computation_graph::first_graph::node_list_iterator poolIt = dataGraph_.node_insert(predIt, allocator_.make_owner<MaxPool>(logger_, **inTensorRes.first, kernelShape, strideX, strideY, padX, padY, name), *inTensorRes.first);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*poolIt)->toString());

    auto currentOp = opsGraph_->get_second_iterator(poolIt);
    controlGraph_.edge_insert(lastOp_, currentOp, ControlFlow());
    lastOp_ = currentOp;

    return poolIt;
}

mv::DataContext::OpListIterator mv::OpModel::concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name)
{

    auto in0TensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*input0).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*in0TensorRes.first)->toString());

    auto in1TensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*input1).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*in1TensorRes.first)->toString());

    computation_graph::first_graph::node_list_iterator concatIt = dataGraph_.node_insert(input0, allocator_.make_owner<Concat>(logger_, **in0TensorRes.first, **in1TensorRes.first, name), *in0TensorRes.first);
    dataGraph_.edge_insert(input1, concatIt, *in1TensorRes.first);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*concatIt)->toString());

    auto currentOp = opsGraph_->get_second_iterator(concatIt);
    controlGraph_.edge_insert(lastOp_, currentOp, ControlFlow());
    lastOp_ = currentOp;

    return concatIt;
}

bool mv::OpModel::addAttr(DataContext::OpListIterator &opIt, const string &name, const Attribute &attr)
{

    if  (attr.getType() == AttrType::TensorType)
    {
        auto resultT = parameterTensors_->insert(allocator_.make_owner<PopulatedTensor>(logger_, (*opIt).getName() + "_" + name, attr.getContent<ConstantTensor>()));
        if (!resultT.second)
            return false;

        bool resultA =  (*opIt).addAttr(name, attr);
        if (!resultA)
        {
            parameterTensors_->erase(*resultT.first);
            return false;
        }

        logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*resultT.first)->toString());

    }
    else
        return (*opIt).addAttr(name, attr);

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
    return dataGraph_.node_end();
}
