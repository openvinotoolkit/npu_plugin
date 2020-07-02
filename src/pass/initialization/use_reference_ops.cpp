#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"

namespace
{
    void useReferenceOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_PASS);

        const auto globalParams = model.getGlobalConfigParams();
        if (!globalParams->hasAttr("ReferenceMode") || !globalParams->get<bool>("ReferenceMode"))
        {
            return;
        }

        mv::OpModel om(model);

        auto convOps = om.getOps("Conv");
        for (auto& origConvOp : convOps)
        {
            const auto origName = origConvOp->getName();

            const auto strides = origConvOp->get<std::array<unsigned short, 2>>("stride");
            const auto padding = origConvOp->get<std::array<unsigned short, 4>>("padding");
            const auto dilationFactor = origConvOp->get<unsigned>("dilationFactor");
            const auto group = origConvOp->get<unsigned>("group");

            const auto origInput = origConvOp->getInputTensor(0);
            const auto origWeights = origConvOp->getInputTensor(1);
            const auto origOutput = origConvOp->getOutputTensor(0);

            auto newConvOutput = om.refConv(
                    origInput, origWeights,
                    strides, padding, dilationFactor, group,
                    mv::DType("Float16"), mv::QuantizationParams({},{},{},{}),
                    origName + "_ref");

            const auto newConvOp = om.getSourceOp(newConvOutput);
            newConvOp->set<bool>("softwareExecuted", true);

            if (origConvOp->hasAttr("opId"))
            {
                const auto origOpId = origConvOp->get<unsigned>("opId");
                newConvOp->set<unsigned>("opId", origOpId);
            }

            if (origOutput->hasAttr("Location"))
            {
                const auto origOutputLocation = origOutput->get<mv::Tensor::MemoryLocation>("Location");
                newConvOutput->set<mv::Tensor::MemoryLocation>("Location", origOutputLocation);
            }

            const auto origConvOutDataFlow = mv::getOutputDataFlow(om, origConvOp, true);
            mv::setOutputDataFlow(om, newConvOutput, origConvOutDataFlow);
        }
    }
}

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(UseReferenceOps)
            .setFunc(useReferenceOpsFcn)
            .setDescription(
                "Replaces HW Operations with reference SW analogues"
            );
    }
}
