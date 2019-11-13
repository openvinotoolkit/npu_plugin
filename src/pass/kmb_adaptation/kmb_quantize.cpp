#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"


static void kmbQuantizeConversionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(KMBQuantizeConversion)
        .setFunc(kmbQuantizeConversionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers between DPUTask-to-UPATask transitions (& vice-versa)."
        );

    }

}

void addQuantizationLayers(mv::OpModel om, std::vector<mv::Data::OpListIterator>& tasks, mv::DType dtypeNeededInInput)
{
    for(auto& task : tasks)
    {
        auto inputFlow = task.leftmostInput();
        auto outputDType = task->getOutputTensor(0)->getDType();
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();
            if(tensorDType != dtypeNeededInInput && !task->hasAttr("mixedPrecision"))
            {
                auto quantize = om.uPATaskQuantize({tensor}, outputDType, tensor->get<mv::QuantizationParams>("quantParams"));
                auto backup = inputFlow;
                ++inputFlow;
                om.undefineFlow(backup);
                om.defineFlow(quantize, task, 0);
                task->setInputTensor(quantize, 0, false);
            }
            else
                ++inputFlow;
        }
    }
}

static void kmbQuantizeConversionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto dpuTasks = om.getOps("DPUTask");
    auto upaTasks = om.getOps("UPATask");

    addQuantizationLayers(om, upaTasks, mv::DType("FP16"));
    //addQuantizationLayers(om, dpuTasks, mv::DType("U8"));
}
