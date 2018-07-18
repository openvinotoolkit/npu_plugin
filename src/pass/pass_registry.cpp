#include "include/mcm/pass/pass_registry.hpp"


mv::pass::PassRegistry& mv::pass::PassRegistry::instance()
{
    
    return static_cast<PassRegistry&>(Registry<PassEntry>::instance());

}
