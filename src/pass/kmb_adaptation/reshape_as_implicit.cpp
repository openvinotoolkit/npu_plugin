#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void reshapeAsImplicitFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ReshapeAsImplicit)
            .setFunc(reshapeAsImplicitFcn)
            .setDescription(
                "Replaces passthrough reshape ops (i.e., output tensor is equivalent to input tensor) with implicit reshapes");
    }
}

void reshapeAsImplicitFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto reshapes = om.getOps("Reshape");

    for(auto& reshape: reshapes)
    {
        auto input = reshape->getInputTensor(0);
        auto input_shape = input->getShape();
        auto output_shape = reshape->get<mv::Shape>("shape");
        auto is_explicit = true;

        // If input & output are both 1D, reshape can be implicit
        auto is_input_flat = (input_shape[0] == input_shape.totalSize() ||
            input_shape[1] == input_shape.totalSize() ||
            input_shape[2] == input_shape.totalSize() ||
            input_shape[3] == input_shape.totalSize());
        auto is_output_flat = (output_shape[0] == output_shape.totalSize() ||
            output_shape[1] == output_shape.totalSize() ||
            output_shape[2] == output_shape.totalSize() ||
            output_shape[3] == output_shape.totalSize());
        if (is_input_flat && is_output_flat)
            is_explicit = false;

        // Skip if explicit
        if (is_explicit)
            continue;

        // Replace reshape op with implicit reshape
        auto outputs = reshape->getOutputTensor();
        auto order = reshape->get<mv::Order>("order");
        auto dtype = input->get<mv::DType>("dType");
        mv::QuantizationParams quantParams = {{}, {}, {}, {}};
        std::string splitStrategy;
        if(reshape->hasAttr("splitStrategy"))
            splitStrategy = reshape->get<std::string>("splitStrategy");
        if(reshape->hasAttr("quantParams"))
            quantParams = reshape->get<mv::QuantizationParams>("quantParams");
        auto outputLocation = reshape->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto opId = reshape->get<unsigned>("opId");
        auto outputFlows = mv::getOutputDataFlow(om, reshape);
        auto implicitReshape = om.implicitReshape(input, output_shape, order, dtype, quantParams);
        om.getSourceOp(implicitReshape)->set<unsigned>("opId", opId);
        implicitReshape->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        if(!splitStrategy.empty())
            om.getSourceOp(implicitReshape)->set<std::string>("splitStrategy", splitStrategy);
        mv::setOutputDataFlow(om, implicitReshape, outputFlows);

        auto parentInputTensor = om.getSourceOp(implicitReshape)->getInputTensor(0);
        if(outputLocation == mv::Tensor::MemoryLocation::OUTPUT)
        {
            //last op
            parentInputTensor->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        }
    }
}
