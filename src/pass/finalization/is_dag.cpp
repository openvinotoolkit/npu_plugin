#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"

static void isDagFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);


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

//NOTE: do not use this pass!!!
void isDagFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Starting IsDag pass");
    mv::ControlModel cm(model);
    if(!cm.isDag())
        throw mv::RuntimeError(cm, "Is not DAG anymore!");
    pass.log(mv::Logger::MessageType::Debug, "Ended IsDag pass");
}
