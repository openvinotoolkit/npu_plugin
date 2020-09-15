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

        for (auto& origConvOp : om.getOps("Conv"))
        {
            const auto origName = origConvOp->getName();

            const auto strides = origConvOp->get<std::array<unsigned short, 2>>("stride");
            const auto padding = origConvOp->get<std::array<unsigned short, 4>>("padding");
            const auto dilationFactor = origConvOp->get<unsigned>("dilationFactor");
            const auto group = origConvOp->get<unsigned>("group");

            const auto origInput = origConvOp->getInputTensor(0);
            const auto weights = origConvOp->getInputTensor(1);
            const auto origOutput = origConvOp->getOutputTensor(0);

            auto newConvOutput = om.refConv(
                    origInput, weights,
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

        // Merge bias operation into RefConv
        for (auto& origBiasOp : om.getOps("Bias"))
        {
            const auto origInput = origBiasOp->getInputTensor(0);
            const auto origBiases = origBiasOp->getInputTensor(1);
            const auto origOutput = origBiasOp->getOutputTensor(0);

            const auto convOp = om.getSourceOp(origInput);
            if (convOp->getOpType() != "RefConv")
            {
                continue;
            }

            const auto biasesData = origBiases->getDoubleData();
            const auto inputShape = origInput->getShape();

            const auto newBiases = om.constant(
                biasesData,
                {1, 1, inputShape[mv::IO_CHANNEL_DIMENSION], 1},
                origBiases->getDType(),
                mv::Order::getZMajorID(4),
                mv::QuantizationParams({},{},{},{}),
                origBiasOp->getName() + "_reshaped");

            if (origBiasOp->hasAttr("opId"))
            {
                const auto origOpId = origBiasOp->get<unsigned>("opId");
                newBiases->set<unsigned>("opId", origOpId);

                const auto newBiasesOp = om.getSourceOp(newBiases);
                newBiasesOp->set<unsigned>("opId", origOpId);
            }

            convOp->addInputTensor(newBiases);
            om.defineFlow(newBiases, convOp, 2);

            auto convOutput = convOp->getOutputTensor(0);
            const auto origOutDataFlow = mv::getOutputDataFlow(om, origBiasOp, true);
            mv::setOutputDataFlow(om, convOutput, origOutDataFlow);

            om.removeOp(om.getSourceOp(origBiases));
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
