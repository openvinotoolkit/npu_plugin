#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

mv::ComputationModel::ComputationModel(Logger &logger) : 
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>()),
logger_(logger)
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>()),
defaultLogger_(allocator_.make_owner<StdOutLogger>(verboseLevel, logTime)),
logger_(*defaultLogger_)
{

}


mv::ComputationModel::ComputationModel(const ComputationModel &other) :
opsGraph_(other.opsGraph_),
dataGraph_(other.dataGraph_),
controlGraph_(other.controlGraph_),
flowTensors_(other.flowTensors_),
parameterTensors_(other.parameterTensors_),
logger_(other.logger_),
input_(other.input_),
output_(other.output_),
lastOp_(other.lastOp_)
{

}

mv::ComputationModel::~ComputationModel()
{

}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && input_ != dataGraph_.node_end() && output_ != dataGraph_.node_end();
}

const mv::Logger &mv::ComputationModel::logger() const
{

    return logger_;

}