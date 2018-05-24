#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

mv::ComputationModel::ComputationModel(Logger &logger) : 
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>()),
logger_(logger),
dataOpEnd_(dataGraph_.node_end()),
dataFlowEnd_(dataGraph_.edge_end()),
controlOpEnd_(controlGraph_.node_end()),
controlFlowEnd_(controlGraph_.edge_end())
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>()),
defaultLogger_(allocator_.make_owner<StdOutLogger>(verboseLevel, logTime)),
logger_(*defaultLogger_),
dataOpEnd_(dataGraph_.node_end()),
dataFlowEnd_(dataGraph_.edge_end()),
controlOpEnd_(controlGraph_.node_end()),
controlFlowEnd_(controlGraph_.edge_end())
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
lastOp_(other.lastOp_),
dataOpEnd_(other.dataOpEnd_),
dataFlowEnd_(other.dataFlowEnd_),
controlOpEnd_(other.controlOpEnd_),
controlFlowEnd_(other.controlFlowEnd_)
{

}

mv::ComputationModel::~ComputationModel()
{

}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && input_ != dataOpEnd_ && output_ != dataOpEnd_;
}

mv::Logger &mv::ComputationModel::logger()
{

    return logger_;

}

/*mv::OpModel mv::ComputationModel::getOpModel()
{
    return OpModel(*this);
}

mv::DataModel mv::ComputationModel::getDataModel()
{
    return DataModel(*this);
}

mv::ControlModel mv::ComputationModel::getControlModel()
{
    return ControlModel(*this);
}*/