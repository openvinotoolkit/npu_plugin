#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(std::string, mv::op::OpEntry)

}

const std::set<std::string> mv::op::OpRegistry::typeTraits_ = 
{
    "executable",   // An op is doing some processing of inputs
    "exposedAPI"       // An op definition call is exposed in CompositionAPI
};

mv::op::OpRegistry& mv::op::OpRegistry::instance()
{
    
    return static_cast<OpRegistry&>(Registry<std::string, OpEntry>::instance());

}
