#include "include/mcm/pass/pass_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(std::string, mv::pass::PassEntry)

}

mv::pass::PassRegistry& mv::pass::PassRegistry::instance()
{
    
    return static_cast<PassRegistry&>(Registry<std::string, PassEntry>::instance());

}
