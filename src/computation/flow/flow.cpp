#include "include/mcm/computation/flow/flow.hpp"

mv::ComputationFlow::ComputationFlow(const std::string &name) :
Element(name)
{

}

/*mv::ComputationFlow::ComputationFlow(mv::json::Value &value):
Element(value)
{

}*/

mv::ComputationFlow::~ComputationFlow()
{
    
}

std::string mv::ComputationFlow::toString() const
{
    return "'" + name_ + "' " + Element::attrsToString_();
}
