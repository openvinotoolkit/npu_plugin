#include "include/mcm/pass/pass_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(pass::PassRegistry, std::string, mv::pass::PassEntry)

}

mv::pass::PassRegistry& mv::pass::PassRegistry::instance()
{
    
    return Registry<PassRegistry, std::string, PassEntry>::instance();

}

void mv::pass::PassRegistry::run(std::string name, ComputationModel& model, TargetDescriptor& targetDescriptor, Element& passDescriptor, Element& output)
{   
    PassEntry* const passPtr = find(name);
    if (passPtr)
        passPtr->run(model, targetDescriptor, passDescriptor, output);
    else
        throw MasterError("PassRegistry", "Invokation of unregistered pass " + name);
}