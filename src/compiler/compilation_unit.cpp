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
    return passManager_.initialize(*model_, targetDescriptor_, compilationDescriptor_);
}

std::pair<std::string, mv::PassGenre> mv::CompilationUnit::runStep()
{
    return passManager_.step();
}

void mv::CompilationUnit::run()
{
    while (!passManager_.completed())
        passManager_.step();
}

bool mv::CompilationUnit::completed() const
{
    return passManager_.completed();
}