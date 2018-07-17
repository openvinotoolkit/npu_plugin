#include "include/mcm/pass/pass_registry.hpp"


mv::base::PassRegistry& mv::base::PassRegistry::instance()
{
    
    return static_cast<PassRegistry&>(base::Registry<PassEntry>::instance());

}
