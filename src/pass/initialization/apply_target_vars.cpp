#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"

static void applyTargetVarsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& cd, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(ApplyTargetVars)
        .setFunc(applyTargetVarsFcn)
        .setDescription(
            "Sets global configuration values based on the target"
        );
    }
}

static void applyTargetVarsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    auto globalConfig = model.getGlobalConfigParams();

    if (!passDesc.hasAttr("Vars"))
    {
        // TODO: Consider emitting a diagnostic
        return;
    }

    auto& vars = passDesc.get<std::vector<mv::Element>>("Vars");

    for (std::size_t idx = 0; idx < vars.size(); ++idx)
    {
        auto& v = vars[idx];
        if (!v.hasAttr("block") || !v.hasAttr("var") || !v.hasAttr("default"))
        {
            // TODO: Consider emitting a diagnostic
            continue;
        }

        if (globalConfig->hasAttr(v.getName()))
        {
            continue;
        }

        int value = v.get<int>("default");

        if (v.get<std::string>("block") == "nce")
        {
            auto it = target.nceDefs().find(v.get<std::string>("var"));
            if (it != target.nceDefs().end())
            {
                value = it->second.totalNumber;
            }
        }
        // TODO: Consider emitting a diagnostic if none of the block cases matched.
        
        globalConfig->set<int>(v.getName(), value);
    }
}
