#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

mv::Logger &mv::ComputationModel::getLogger(Logger::VerboseLevel verboseLevel, bool logTime)
{
    static StdOutLogger logger(verboseLevel, logTime);
    return logger;
}

mv::ComputationModel::ComputationModel(Logger &logger) : 
logger_(logger)
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
logger_(getLogger(verboseLevel, logTime))
{

}

const mv::OpListIterator mv::ComputationModel::input(const Shape &shape, DType dType, Order order, const string &name)
{

    string inputName = name;

    if (inputName.empty())
        inputName = "0";

    input_ = ops_graph.node_insert(allocator_.make_owner<Input>(logger_, inputName, shape, dType, order));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());

    return input_;

}

const mv::OpListIterator mv::ComputationModel::output(OpListIterator &predecessor, const string &name)
{

    string outputName = name;

    if (outputName.empty())
        outputName = "0";

    auto inTensor = predecessor->getOutput();
    output_ = ops_graph.node_insert(predecessor, allocator_.make_owner<Output>(logger_, outputName, inTensor), inTensor);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());

    return output_;

}

mv::OpListIterator mv::ComputationModel::convolutional(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, const string &name)
{

    string convName = name;

    if (convName.empty())
        convName = "0";

    auto inTensor = predecessor->getOutput();
    OpListIterator conv = ops_graph.node_insert(predecessor, allocator_.make_owner<Conv>(logger_, convName, inTensor, weights, strideX, strideY), inTensor);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + conv->toString());

    return conv;
}

bool mv::ComputationModel::addAttr(OpListIterator &op, const string &name, const Attribute &attr)
{
    return op->addAttr(name, attr);
}

const mv::Logger &mv::ComputationModel::logger() const
{

    return logger_;

}