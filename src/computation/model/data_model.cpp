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

bool mv::DataModel::isValid() const
{
    return ComputationModel::isValid();
}

mv::DataListIterator mv::DataModel::getInput()
{
    return input_->leftmost_output();
}

mv::DataListIterator mv::DataModel::getOutput()
{
    return output_->leftmost_input();
}