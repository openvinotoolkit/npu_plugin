#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void permuteAsImplicitFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(PermuteAsImplicit)
            .setFunc(permuteAsImplicitFcn)
            .setDescription(
                "Replaces passthrough permute ops (i.e., output tensor is equivalent to input tensor) with implicit permutes");
    }
}

void permuteAsImplicitFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto permutes = om.getOps("Permute");

    for(auto& permute: permutes)
    {
        auto input = permute->getInputTensor(0);
        auto output = permute->getOutputTensor(0);
        auto input_shape = input->getShape();
        auto output_shape = output->getShape();
        auto is_explicit = true;

        // If input & output are both 1D, permute can be implicit
        if (input_shape.isFlat() && output_shape.isFlat())
            is_explicit = false;

        // Skip if explicit
        if (is_explicit)
            continue;

        // Replace permute op with implicit permute
        auto dtype = input->get<mv::DType>("dType");
        mv::QuantizationParams quantParams = {{}, {}, {}, {}};
        std::string splitStrategy;
        if(permute->hasAttr("splitStrategy"))
            splitStrategy = permute->get<std::string>("splitStrategy");
        if(permute->hasAttr("quantParams"))
            quantParams = permute->get<mv::QuantizationParams>("quantParams");
        auto outputLocation = permute->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto opId = permute->get<unsigned>("opId");
        auto outputFlows = mv::getOutputDataFlow(om, permute);
        auto implicitPermute = om.implicitPermute(input, output_shape, dtype, quantParams);
        om.getSourceOp(implicitPermute)->set<unsigned>("opId", opId);
        implicitPermute->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        if(!splitStrategy.empty())
            om.getSourceOp(implicitPermute)->set<std::string>("splitStrategy", splitStrategy);
        mv::setOutputDataFlow(om, implicitPermute, outputFlows);

        auto parentInputTensor = om.getSourceOp(implicitPermute)->getInputTensor(0);
        if(outputLocation == mv::Tensor::MemoryLocation::OUTPUT)
        {
            //last op
            parentInputTensor->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        }
    }
}
