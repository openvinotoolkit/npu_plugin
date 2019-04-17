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
            "This pass extends all quantization params to the size of output channel, preparing them for serialization."
        );
    }
}

void extendQuantizationParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        size_t outputChannels = 0;
        if (opIt->getOutputTensor().size() > 0)
        {
            auto output = opIt->getOutputTensor(0);
            if (output->hasAttr("quantizationParams"))
            {
                outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
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
                if (opIt->getOutputTensor().size() == 0 || outputChannels == 0)
                    outputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto inputQuantization = input->get<mv::QuantizationParams>("quantizationParams");
                inputQuantization.extendParamsToOutputChannelSize(outputChannels);
                input->set<mv::QuantizationParams>("quantizationParams", inputQuantization);
            }
        }

    }
}
