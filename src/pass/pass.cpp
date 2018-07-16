#include "include/mcm/pass/pass.hpp"

constexpr mv::pass::Pass::Pass()
{
    //PassRegistry::passRegistry_[uid_] = "pass1";
}

bool mv::pass::Pass::run(ComputationModel &model)
{
    return true;
}
