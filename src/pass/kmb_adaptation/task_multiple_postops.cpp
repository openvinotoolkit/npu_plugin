#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"

static void taskMultiplePostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TaskMultiplePostOps)
        .setFunc(taskMultiplePostOpsFcn)
        .setDescription(
            "Resolve DPUTasks with multiple postops."
        );
    }
}

template<typename T>
void removeFromVector(std::vector<T>& vector, const T& value)
{
    vector.erase(std::remove(vector.begin(), vector.end(), value), vector.end());
}

void resolveRelu(mv::Data::OpListIterator& opIt)
{
    if (!opIt->hasAttr("PPETask"))
        return;

    auto ppeFF = opIt->get<mv::PPETask>("PPETask").getFixedFunction();

    auto newPpeFF = mv::PPEFixedFunction(
                        ppeFF.getLReluMult(),
                        ppeFF.getLReluShift(),
                        std::max(ppeFF.getLowClamp(), 0),
                        ppeFF.getHighClamp()
                    );

    auto layers = ppeFF.getLayers();
    removeFromVector<mv::PPELayerType>(layers, mv::PPELayerType("Relu"));

    for (auto& layer : layers)
        newPpeFF.addLayer(layer);

    opIt->set<mv::PPETask>("PPETask", mv::PPETask(newPpeFF));
}

void taskMultiplePostOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto contains = [](const std::vector<std::string>& postOps, const std::string& postOp) {
        return std::find(postOps.begin(), postOps.end(), postOp) != postOps.end();
    };

    auto dpuTasks = om.getOps("DPUTask");

    for (auto& dpuTask : dpuTasks)
    {
        if (!dpuTask->hasAttr("postOpTypes"))
            continue;

        auto postOps = dpuTask->get<std::vector<std::string>>("postOpTypes");

        if (postOps.size() < 2)
            continue;

        if (contains(postOps, "Relu"))
        {
            resolveRelu(dpuTask);
            removeFromVector<std::string>(postOps, "Relu");
        }

        dpuTask->set<std::vector<std::string>>("postOpTypes", postOps);
    }
}
