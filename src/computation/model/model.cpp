#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

/*mv::Logger &mv::ComputationModel::getLogger(Logger::VerboseLevel verboseLevel, bool logTime)
{
    static StdOutLogger logger(verboseLevel, logTime);
    return logger;
}*/

mv::ComputationModel::ComputationModel(Logger &logger) : 
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedModelTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedModelTensor>, TensorOrderComparator>()),
logger_(logger)
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedModelTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedModelTensor>, TensorOrderComparator>()),
defaultLogger_(allocator_.make_owner<StdOutLogger>(verboseLevel, logTime)),
logger_(*defaultLogger_)
//ComputationModel(*defaultLogger_.get())
{

}

const mv::OpListIterator mv::ComputationModel::input(const Shape &shape, DType dType, Order order, const string &name)
{

    string inputName = name;

    if (inputName.empty())
        inputName = "0";

    input_ = ops_graph_.node_insert(allocator_.make_owner<Input>(logger_, inputName, shape, dType, order));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());

    return input_;

}

const mv::OpListIterator mv::ComputationModel::output(OpListIterator &predecessor, const string &name)
{

    string outputName = name;

    if (outputName.empty())
        outputName = "0";

    auto inTensorIt = flowTensors_->insert(allocator_.make_owner<UnpopulatedModelTensor>(predecessor->getOutput()));
    output_ = ops_graph_.node_insert(predecessor, allocator_.make_owner<Output>(logger_, outputName, **inTensorIt.first), *inTensorIt.first);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());

    return output_;

}

mv::OpListIterator mv::ComputationModel::conv2D(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name)
{

    string convName = name;

    if (convName.empty())
        convName = "0";

    auto inTensorIt = flowTensors_->insert(allocator_.make_owner<UnpopulatedModelTensor>(predecessor->getOutput()));
    OpListIterator conv = ops_graph_.node_insert(predecessor, allocator_.make_owner<Conv>(logger_, convName, **inTensorIt.first, weights, strideX, strideY, padX, padY), *inTensorIt.first);
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + conv->toString());

    return conv;
}

bool mv::ComputationModel::addAttr(OpListIterator &op, const string &name, const Attribute &attr)
{


    if  (attr.getType() == AttrType::TensorType)
    {
        auto resultT = parameterTensors_->insert(allocator_.make_owner<PopulatedModelTensor>(logger_, op->getName() + "_" + name, attr.getContent<ConstantTensor>()));
        if (!resultT.second)
            return false;

        bool resultA =  op->addAttr(name, attr);
        if (!resultA)
        {
            parameterTensors_->erase(*resultT.first);
            return false;
        }

    }
    else
        return op->addAttr(name, attr);

    return false;

}

bool mv::ComputationModel::isValid() const
{
    return !ops_graph_.disjoint() && input_ != ops_graph_.node_end() && output_ != ops_graph_.node_end();
}

const mv::Logger &mv::ComputationModel::logger() const
{

    return logger_;

}