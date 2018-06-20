#include "include/mcm/pass/deploy_pass.hpp"

mv::Logger& mv::pass::DeployPass::logger_ = mv::ComputationModel::logger();

mv::pass::DeployPass::DeployPass(OStream& ostream) :
ostream_(ostream)
{

}

mv::pass::DeployPass::~DeployPass()
{

}

bool mv::pass::DeployPass::run(ComputationModel& model)
{

    if (!ostream_.open())
        return false;
    bool result = run_(model);
    ostream_.close();

    return result;

}