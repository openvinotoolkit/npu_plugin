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
        std::size_t id = 0;
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();

            // NOTE: Maybe here a check for mixed precision should be added
            if(!tensor->isPopulated() && tensorDType != dtypeNeededInInput)
            {
                auto quantize = om.uPATaskQuantize({tensor}, outputDType,
                            tensor->get<mv::QuantizationParams>("quantParams"), "Quantize" + task->getName() + std::to_string(id));
                quantize->set<std::string>("splitStrategy",
                            tensor->get<std::string>("splitStrategy"));
                auto quantizeOp = om.getSourceOp(quantize);
                quantizeOp->set<unsigned>("opId", task->get<unsigned>("opId"));
                auto backup = inputFlow;
                auto slot = backup->get<size_t>("sinkInput");
                ++inputFlow;
                om.undefineFlow(backup);
                task->setInputTensor(quantize, slot, false);
                om.defineFlow(quantize, task, slot);
                id++;
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

    auto dpuTasks = om.getOps("DPUTask");
    auto upaTasks = om.getOps("UPATask");
    auto concats = om.getOps("Concat");
    auto implicitConcats = om.getOps("ImplicitConcat");
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();

    concats.insert(concats.end(), implicitConcats.begin(), implicitConcats.end());
    addQuantizationLayers(om, upaTasks, mv::DType("Float16"));

    bool DPUTasksinSW = globalParams->hasAttr("DPUTasksinFloat") ? globalParams->get<bool>("DPUTasksinFloat") : false;
    if (!DPUTasksinSW)
    {
        addQuantizationLayers(om, dpuTasks, mv::DType("UInt8"));

        // NOTE: For now let's do all the concats in UInt8
        // For the future, it might be good to optimize this to
        // insert the smallest number possible of Quantization Layers.
        addQuantizationLayers(om, concats, mv::DType("UInt8"));
    }
}
