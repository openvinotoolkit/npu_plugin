#include "include/mcm/computation/flow/flow.hpp"

mv::ComputationFlow::ComputationFlow(const std::string &name) :
ComputationElement(name)
{

}

mv::ComputationFlow::ComputationFlow(mv::json::Value &value):
ComputationElement(value)
{

}

mv::ComputationFlow::~ComputationFlow()
{
    
}

std::string mv::ComputationFlow::toString() const
{
    return "'" + name_ + "' " + ComputationElement::toString();
}
