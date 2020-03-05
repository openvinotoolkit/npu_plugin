#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void updateImplicitLayersQuantizationParamsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateImplicitLayersLocationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(UpdateImplicitLayersQuantizationParams)
            .setFunc(updateImplicitLayersQuantizationParamsFcn)
            .setDescription(
                "Update Quantization Params for Implicit Layers output after input layers have been quantized");
    }
    namespace pass
    {
        MV_REGISTER_PASS(UpdateImplicitLayersLocation)
            .setFunc(updateImplicitLayersLocationParamsFcn)
            .setDescription(
                "Update Location of Implicit Ops");
    }
}

void updateImplicitLayersQuantizationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto sortedOps = om.topologicalSort();
    for(auto opIt : sortedOps)
    {
         std::string opType = opIt->getOpType();

        if (opIt->getOpType() ==  "ImplicitConcat" || opIt->getOpType() ==  "ImplicitReshape" || opIt->getOpType() ==  "ImplicitPermute" || opIt->getOpType() ==  "Copy" || opIt->getOpType() ==  "Slice"
            || opIt->getOpType() ==  "Crop" || opIt->getOpType() ==  "Align")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            if (input->hasAttr("quantParams"))
            {
                mv::QuantizationParams &inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                opIt->set<mv::QuantizationParams>("quantParams", inputQuantization);
                output->set<mv::QuantizationParams>("quantParams", inputQuantization);
            }
        }
    }
}

void updateImplicitLayersLocationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto sortedOps = om.topologicalSort();
    for(auto opIt : sortedOps)
    {
         std::string opType = opIt->getOpType();

        if (opType ==  "Slice" || opType ==  "Crop" ||
            opType == "ImplicitReshape" || opType == "ImplicitPermute")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);

            auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", inputMemoryLocation);
        }

    }
}
