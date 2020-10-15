#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void arrangeKmbExecutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ArrangeKmbExecution)
        .setFunc(arrangeKmbExecutionFcn)
        .setDescription(
            ""
        );

    }

}

// This pass has one main duty
// 1) Put the stages
// Point 1) is trivial for now (just 1 stage), but will be probably updated when Pat completes his analysis

void arrangeKmbExecutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Starting arrange Kmb execution");

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // Point 2)
    auto stage = cm.addStage();
    cm.addToStage(stage, om.getOutput());

    pass.log(mv::Logger::MessageType::Debug, "Exiting arrange Kmb execution");

}

