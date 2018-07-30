#include "include/mcm/pass/pass_registry.hpp"

mv::pass::RutimeError::RutimeError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}


mv::pass::PassRegistry& mv::pass::PassRegistry::instance()
{
    
    return static_cast<PassRegistry&>(Registry<PassEntry>::instance());

}
