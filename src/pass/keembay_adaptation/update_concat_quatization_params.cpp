#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void updateConcatQuantizationParamsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(UpdateConcatQuantizationParams)
            .setFunc(updateConcatQuantizationParamsFcn)
            .setDescription(
                "Update Quantization Params for Concat output after input layers have been quantized");
    }
}

void updateConcatQuantizationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
         std::string opType = opIt->getOpType();

        if (opIt->getOpType() ==  "ImplicitConcat")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            if (input->hasAttr("quantParams"))
            {
                mv::QuantizationParams &inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                outputQuantization.quantize(inputQuantization.getShift(), inputQuantization.getMult());
            }

        }
    }
}
