#include "include/mcm/pass/transform_pass.hpp"

mv::Logger& mv::pass::TransformPass::logger_ = mv::ComputationModel::logger();

mv::pass::TransformPass::TransformPass(const string& name) :
name_(name)
{

}

mv::pass::TransformPass::~TransformPass()
{

}

bool mv::pass::TransformPass::run(ComputationModel& model)
{

    bool result = run_(model);
    return result;

}