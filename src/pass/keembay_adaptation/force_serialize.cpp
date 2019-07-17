#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void forceSerializeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ForceSerialize)
            .setFunc(forceSerializeFcn)
            .setDescription(
                "Serialize the DPU tasks for graphs with parallel paths");
    }
}

void forceSerializeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto removeNonEssentialOps = [] (std::vector<mv::Data::OpListIterator>& list)
    {
        list.erase(std::remove_if(list.begin(),
                                list.end(),
                                [](mv::Data::OpListIterator it)
                                {
                                    return (it->getOpType() != "DPUTask") && (it->getOpType() != "Input");
                                }),
                                list.end());
    };

    // This *NEEDS* to be based on the order of the Operational model
    auto sortedOps = om.topologicalSort(true);
    removeNonEssentialOps(sortedOps);

    for (size_t i = 0; i < sortedOps.size() - 1; i++)
    {
        pass.log(mv::Logger::MessageType::Debug, " sortedOps[" +  std::to_string(i) + "] = " + sortedOps[i]->getName() );;
        if (!(cm.pathExists(cm.switchContext(sortedOps[i]), cm.switchContext(sortedOps[i+1]))
            || om.pathExists(sortedOps[i], sortedOps[i+1])
            || cm.pathExists(cm.switchContext(sortedOps[i+1]), cm.switchContext(sortedOps[i]))
            || om.pathExists(sortedOps[i+1], sortedOps[i])))
        {
            pass.log(mv::Logger::MessageType::Debug,
                "FORCE SERIALIZE: adding edge from " + sortedOps[i]->getName() + " to " +
                sortedOps[i+1]->getName());

            cm.defineFlow(sortedOps[i], sortedOps[i+1]);
        }
    }

}
