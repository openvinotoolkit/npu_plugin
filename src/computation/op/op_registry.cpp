#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(std::string, mv::op::OpEntry)

}

mv::op::OpRegistry& mv::op::OpRegistry::instance()
{
    
    return static_cast<OpRegistry&>(Registry<std::string, OpEntry>::instance());

}
