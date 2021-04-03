#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void resampleAsImplicitFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ResampleAsImplicit)
            .setFunc(resampleAsImplicitFcn)
            .setDescription(
                "Replaces passthrough resample ops with implicit resample");
    }
}

void resampleAsImplicitFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto resamples = om.getOps("Resample");

    for(auto& resample: resamples)
    {
        // Skip if explicit
        if (!(resample->hasAttr("isImplicit") && resample->get<bool>("isImplicit")))
            continue;

        auto input = resample->getInputTensor(mv::IO_TENSOR_INPUT);
        auto output = resample->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        auto input_shape = input->getShape();
        auto output_shape = output->getShape();

        // Replace resample op with implicit resample
        std::string splitStrategy;
        if(resample->hasAttr("splitStrategy"))
            splitStrategy = resample->get<std::string>("splitStrategy");
        auto quantParams = output->getQuantParams();
        mv::Tensor::MemoryLocation outputLocation;
        if(input->hasAttr("Location"))
            outputLocation = input->get<mv::Tensor::MemoryLocation>("Location");
        auto opId = resample->get<unsigned>("opId");
        auto outputFlows = mv::getOutputDataFlow(om, resample);
        auto implicitResample = om.implicitResample("", input, output_shape);
        implicitResample->setQuantParams(quantParams);
        om.getSourceOp(implicitResample)->set<unsigned>("opId", opId);
        // Store input shapes, used later to compute SEP table offsets
        om.getSourceOp(implicitResample)->set<mv::Shape>("originalShape", input_shape);
        implicitResample->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        if(!splitStrategy.empty())
            om.getSourceOp(implicitResample)->set<std::string>("splitStrategy", splitStrategy);
        mv::setOutputDataFlow(om, implicitResample, outputFlows);

        auto parentInputTensor = om.getSourceOp(implicitResample)->getInputTensor(mv::IO_TENSOR_INPUT);
        if(outputLocation == mv::Tensor::MemoryLocation::OUTPUT)
        {
            //last op
            parentInputTensor->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        }
    }
}
