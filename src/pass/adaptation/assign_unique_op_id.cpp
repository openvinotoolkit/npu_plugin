#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void assignUniqueOpIdFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

void assignUniqueOpIdFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    unsigned currentId = 0;
    std::string currentIdLabel("opId");

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
        operationIt->set<unsigned>(currentIdLabel, currentId++);
}
