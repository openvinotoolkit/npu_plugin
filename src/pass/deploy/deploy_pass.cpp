#include "include/mcm/pass/deploy/deploy_pass.hpp"

mv::pass::DeployPass::DeployPass(Logger& logger, OStream& ostream) :
logger_(logger),
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