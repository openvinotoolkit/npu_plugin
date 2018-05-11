#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

mv::ComputationModel::ComputationModel(Logger &logger) : 
ops_graph_(allocator_.make_owner<computation_graph>(computation_graph())),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>()),
logger_(logger)
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ops_graph_(allocator_.make_owner<computation_graph>(computation_graph())),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>()),
defaultLogger_(allocator_.make_owner<StdOutLogger>(verboseLevel, logTime)),
logger_(*defaultLogger_)
{

}

mv::ComputationModel::ComputationModel(const ComputationModel &other) :
ops_graph_(other.ops_graph_),
flowTensors_(other.flowTensors_),
parameterTensors_(other.parameterTensors_),
logger_(other.logger_),
input_(other.input_),
output_(other.output_)
{

}

mv::ComputationModel::~ComputationModel()
{

}

bool mv::ComputationModel::isValid() const
{
    return !ops_graph_->disjoint() && input_ != ops_graph_->node_end() && output_ != ops_graph_->node_end();
}

const mv::Logger &mv::ComputationModel::logger() const
{

    return logger_;

}