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

            // NOTE: Maybe here a check for mixed precision should be added
            if(!tensor->isPopulated() && tensorDType != dtypeNeededInInput)
            {
                auto quantize = om.uPATaskQuantize({tensor}, outputDType,
                            tensor->get<mv::QuantizationParams>("quantParams"), "Quantize" + task->getName());
                quantize->set<std::string>("splitStrategy",
                            tensor->get<std::string>("splitStrategy"));
                auto quantizeOp = om.getSourceOp(quantize);
                quantizeOp->set<unsigned>("opId", task->get<unsigned>("opId"));
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

static void kmbQuantizeConversionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    //Note: We add eltwise support for fp16 as well...
    auto dpuTasks = om.getOps("DPUTask");
    auto upaTasks = om.getOps("UPATask");

    addQuantizationLayers(om, upaTasks, mv::DType("Float16"));
    addQuantizationLayers(om, dpuTasks, mv::DType("UInt8"));
}
