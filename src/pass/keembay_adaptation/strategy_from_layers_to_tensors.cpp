#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void strategyLayersToTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StrategyLayersToTensors)
        .setFunc(strategyLayersToTensors)
        .setDescription(
            "Extend strategies from ops to tensors"
        );
    }
}

void strategyLayersToTensors(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        std::string opType = layer->getOpType();
        if (opType == "DPUTask")
        {
            auto opStrategy = layer->get<std::string>("splitStrategy");
            auto outputTensor = layer->getOutputTensor(0);
            outputTensor->set<std::string>("splitStrategy", opStrategy);
            unsigned n = layer->inputSlots();
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = layer->getInputTensor(i);
                inputTensor->set<std::string>("splitStrategy", opStrategy);
            }
        }
     }
    return;
}
