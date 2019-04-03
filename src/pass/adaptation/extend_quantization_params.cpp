#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

static void extendQuantizationParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ExtendQuantizationParams)
        .setFunc(extendQuantizationParams)
        .setDescription(
            "This pass assigns an unique ID to each op in the graph."
        );
    }
}

void extendQuantizationParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        size_t outputChannels;
        if (opIt->getOutputTensor().size() > 0)
        {
            auto output = opIt->getOutputTensor(0);
            outputChannels = output->getShape()[2];
            if (output->hasAttr("quantizationParams"))
            {
                auto outputQuantization = output->get<mv::QuantizationParams>("quantizationParams");
                outputQuantization.extendParamsToOutputChannelSize(outputChannels);
                output->set<mv::QuantizationParams>("quantizationParams", outputQuantization);
            }

        }
        if (opIt->getInputTensor().size() > 0)
        {
            auto input = opIt->getInputTensor(0);
            if (input->hasAttr("quantizationParams"))
            {
                if (opIt->getOutputTensor().size() == 0)
                    outputChannels = input->getShape()[2];
                auto inputQuantization = input->get<mv::QuantizationParams>("quantizationParams");
                inputQuantization.extendParamsToOutputChannelSize(outputChannels);
                input->set<mv::QuantizationParams>("quantizationParams", inputQuantization);
            }
        }

    }
}
