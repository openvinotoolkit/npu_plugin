#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void assignUniqueOpIdFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AssignUniqueOpId)
        .setFunc(assignUniqueOpIdFcn)
        .setDescription(
            "This pass assigns an unique ID to each op in the graph."
        );
    }
}

void assignUniqueOpIdFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    unsigned currentId = 1;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
        operationIt->set<unsigned>("opId", currentId++);
}
