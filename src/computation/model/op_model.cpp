#include "include/fathom/computation/model/op_model.hpp"

mv::OpModel::OpModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::OpModel::OpModel(Logger &logger) :
ComputationModel(logger)
{

}

mv::OpListIterator mv::OpModel::input(const Shape &shape, DType dType, Order order, const string &name)
{

    string inputName = name;

    if (inputName.empty())
        inputName = "0";

    input_ = ops_graph_->node_insert(allocator_.make_owner<Input>(logger_, inputName, shape, dType, order));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*input_).toString());

    return input_;

}

mv::OpListIterator mv::OpModel::output(OpListIterator &predIt, const string &name)
{

    string outputName = name;

    if (outputName.empty())
        outputName = "0";

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*predIt).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    output_ = ops_graph_->node_insert(predIt, allocator_.make_owner<Output>(logger_, outputName, **inTensorRes.first), *inTensorRes.first);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*output_).toString());

    return output_;

}

mv::OpListIterator mv::OpModel::conv2D(OpListIterator &predIt, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    string convName = name;

    if (convName.empty())
        convName = "0";

    auto inTensorRes = flowTensors_->insert(allocator_.make_owner<UnpopulatedTensor>((*predIt).getOutput()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*inTensorRes.first)->toString());

    OpListIterator convIt = ops_graph_->node_insert(predIt, allocator_.make_owner<Conv>(logger_, convName, **inTensorRes.first, weights, strideX, strideY, padX, padY), *inTensorRes.first);
    auto resultT = parameterTensors_->insert(allocator_.make_owner<PopulatedTensor>(logger_, (*convIt).getName() + "_" + "weights", (*convIt).getAttr("weights").getContent<ConstantTensor>()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*resultT.first)->toString());
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*convIt).toString());

    return convIt;
}

bool mv::OpModel::addAttr(OpListIterator &opIt, const string &name, const Attribute &attr)
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

mv::OpListIterator mv::OpModel::getInput()
{
    return input_;
}

mv::OpListIterator mv::OpModel::getOutput()
{
    return output_;
}