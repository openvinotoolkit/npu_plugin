#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/myriadx/nce1.hpp"

static void convertToTaskGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ConvertToTaskGraph)
        .setFunc(convertToTaskGraph)
        .setDescription(
            "This pass converts a graph of operations into a graph of tasks."
        );
    }
}

void convertToTaskGraph(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    // TODO
}
