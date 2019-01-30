#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/myriadx/nce1.hpp"

static void assignUniqueTaskId(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AssignUniqueTaskId)
        .setFunc(assignUniqueTaskId)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "This pass assigns an unique ID to each node in the graph."
        );
    }
}

void assignUniqueTaskId(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    mv::OpModel om(model);

    unsigned currentId = 0;
    std::string currentIdLabel("taskId");

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
        operationIt->set<unsigned>(currentIdLabel, currentId++);
}
