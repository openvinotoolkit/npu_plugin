#include "include/mcm/compiler/compilation_unit.hpp"

mv::CompilationUnit::CompilationUnit(mv::Logger::VerboseLevel verboseLevel, bool logTime) :
model_(new OpModel(verboseLevel, logTime))
{

}

mv::CompilationUnit::~CompilationUnit()
{
    delete model_;
}

mv::PassManager& mv::CompilationUnit::passManager()
{
    return passManager_;
}

mv::TargetDescriptor& mv::CompilationUnit::targetDescriptor()
{
    return targetDescriptor_;
}

mv::json::Object& mv::CompilationUnit::compilationDescriptor()
{
    return compilationDescriptor_;
}

mv::CompositionalModel& mv::CompilationUnit::model()
{
    return *model_;
}

bool mv::CompilationUnit::initialize()
{

    if (!passManager_.initialize(*model_, targetDescriptor_, compilationDescriptor_))
        return false;
    // Initialize resouces
    for (auto it = targetDescriptor_.memoryDefs().begin(); it != targetDescriptor_.memoryDefs().end(); ++it)
    {
        mv::DataModel dm(*model_);
        dm.addAllocator(it->first, it->second.size);
    }

    return true;

}

mv::json::Object mv::CompilationUnit::runStep()
{
    return passManager_.step();
}

mv::json::Object mv::CompilationUnit::run()
{
    json::Object output;
    while (!passManager_.completed())
        output = passManager_.step();
    return output;
}

bool mv::CompilationUnit::completed() const
{
    return passManager_.completed();
}