#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void transitiveReductionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passArg, mv::json::Object&);


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

void transitiveReductionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passArg, mv::json::Object&)
{

    pass.log(mv::Logger::MessageType::Debug, "Starting transitive reduction pass");
    mv::ControlModel cm(model);

    auto filter = passArg.get<std::string>("filter");
    cm.transitiveReduction(filter);
    pass.log(mv::Logger::MessageType::Debug, "Ended transitive reduction pass");
}
