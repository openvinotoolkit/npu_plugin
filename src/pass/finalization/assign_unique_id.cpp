#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/myriadx/nce1.hpp"

static void assignUniqueId(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AssignUniqueId)
        .setFunc(assignUniqueId)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass assigns an unique ID to each node in the graph. The pass can be run multiple times with multiple labels"
        );
    }
}

void assignUniqueId(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    unsigned currentId = 0;
    std::string currentIdLabel("testIdLabel"); // TODO: Must be taken from pass parameters

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
        operationIt->set<unsigned>(currentIdLabel, currentId++);
}
