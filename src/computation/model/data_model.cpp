#include "include/fathom/computation/model/data_model.hpp"

/*mv::DataModel::DataModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::DataModel::DataModel(Logger &logger) :
ComputationModel(logger)
{

}*/

mv::DataModel::DataModel(const ComputationModel &other) :
ComputationModel(other)
{

}

mv::DataContext::OpListIterator mv::DataModel::switchContext(ControlContext::OpListIterator &other)
{
    return opsGraph_->get_first_iterator(other);
}

bool mv::DataModel::isValid() const
{
    return ComputationModel::isValid();
}

mv::DataContext::FlowSiblingIterator mv::DataModel::getInput()
{
    return input_.leftmostOutput();
}

mv::DataContext::FlowSiblingIterator mv::DataModel::getOutput()
{
    return output_.leftmostInput();
}

mv::DataContext::FlowListIterator mv::DataModel::flowEnd()
{
    return dataFlowEnd_;
}