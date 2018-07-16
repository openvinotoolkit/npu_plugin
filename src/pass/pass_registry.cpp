#include "include/mcm/pass/pass_registry.hpp"


mv::pass::PassRegistry *mv::pass::PassRegistry::instance()
{
    return nullptr;
    //return static_cast<PassRegistry*>(base::Singleton::instance());

}
