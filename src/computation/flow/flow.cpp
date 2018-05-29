#include "include/fathom/computation/flow/flow.hpp"

mv::ComputationFlow::ComputationFlow(const Logger &logger, const string &name) :
ComputationElement(logger, name)
{

}

mv::ComputationFlow::~ComputationFlow()
{
    
}

mv::string mv::ComputationFlow::toString() const
{
    return "'" + name_ + "' " + ComputationElement::toString();
}