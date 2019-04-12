#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void isDagFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(IsDAG)
        .setFunc(isDagFcn)
        .setDescription(
            ""
        );

    }

}

void isDagFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    pass.log(mv::Logger::MessageType::Debug, "Starting IsDag pass");
    mv::ControlModel cm(model);
    if(!cm.isDag())
        throw "Is not DAG anymore!";
    pass.log(mv::Logger::MessageType::Debug, "Ended IsDag pass");
}
