#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void transitiveReductionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passArg, mv::Element&);


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TransitiveReduction)
        .setFunc(transitiveReductionFcn)
        .setDescription(
            ""
        );

    }

}

void transitiveReductionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passArg, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Starting transitive reduction pass");
    mv::ControlModel cm(model);

    std::string filter;
    if(passArg.hasAttr("filter"))
        filter = passArg.get<std::string>("filter");

    cm.transitiveReduction(filter);
    pass.log(mv::Logger::MessageType::Debug, "Ended transitive reduction pass");
}
