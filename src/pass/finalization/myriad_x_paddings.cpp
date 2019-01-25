#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/myriadx/nce1.hpp"

static void myriadXPaddings(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MyriadXPaddings)
        .setFunc(myriadXPaddings)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass pads all tensors in the network to be compatible with MX"
        );
    }
}

//ASSUMPTION 1: This pass must be executed after the Mark Hardware Convolution pass.
//REASON: There is no need to pad tensors not involved in HW operations at all.
void myriadXPaddings(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Nce1 nce;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        if(!operationIt->hasAttr("NCE1_Compatible"))
            continue;
        if(!operationIt->get<int>("NCE1_Compatible"))
            continue;

        bool isConv = operationIt->getOpType() == "Conv";

        operationIt->getInputTensor(0)->setOrder(mv::Order("HCW"));
        operationIt->getOutputTensor(0)->setOrder(mv::Order("HCW"));
        // operationIt->getInputTensor(0)->setOrder(mv::Order("CHW"));
        // operationIt->getOutputTensor(0)->setOrder(mv::Order("CHW"));

        auto input_tensor = operationIt->getInputTensor(0);
        auto input_tensor_dimension = input_tensor->getShape();

        auto output_tensor = operationIt->getOutputTensor(0);
        auto output_tensor_dimension = output_tensor->getShape();

        size_t input_width = input_tensor_dimension[0];
        size_t input_height = input_tensor_dimension[1];

        size_t actual_input_height = nce.computeActualInputHeight(input_height);
        size_t actual_input_width = nce.computeActualInputWidth(input_width);

        size_t output_width = output_tensor_dimension[0];
        size_t output_height = output_tensor_dimension[1];

        size_t actual_output_width = nce.computeActualOutputWidth(output_width);
        size_t actual_output_height = nce.computerActualOutputHeight(output_height);

        //God please forgive me for the magic numbers
        std::vector<size_t> input_tensor_paddings(3);
        std::vector<size_t> output_tensor_paddings(3);

        input_tensor_paddings[0] = actual_input_width - input_width;
        input_tensor_paddings[1] = actual_input_height - input_height;
        input_tensor_paddings[2] = 0;

        output_tensor_paddings[0] = actual_output_width - output_width;
        output_tensor_paddings[1] = actual_output_height - output_height;
        output_tensor_paddings[2] = 0;

        if(input_tensor->hasAttr("NCE1_Paddings"))
        {
            //Simple rule: greatest padding wins
            std::vector<size_t> existing_input_tensor_paddings = input_tensor->get<std::vector<size_t>>("NCE1_Paddings");
            input_tensor->erase("NCE1_Paddings");
            for(unsigned i = 0; i < existing_input_tensor_paddings.size();++i)
                input_tensor_paddings[i] = std::max(input_tensor_paddings[i], existing_input_tensor_paddings[i]);
        }

        if(output_tensor->hasAttr("NCE1_Paddings"))
        {
            //Simple rule: greatest padding wins
            std::vector<size_t> existing_output_tensor_paddings = input_tensor->get<std::vector<size_t>>("NCE1_Paddings");
            output_tensor->erase("NCE1_Paddings");
            for(unsigned i = 0; i < existing_output_tensor_paddings.size();++i)
                output_tensor_paddings[i] = std::max(output_tensor_paddings[i], existing_output_tensor_paddings[i]);
        }

        input_tensor->set<std::vector<size_t>>("NCE1_Paddings", input_tensor_paddings);
        output_tensor->set<std::vector<size_t>>("NCE1_Paddings", output_tensor_paddings);

        if(isConv)
        {
            auto weight_tensor = operationIt->getInputTensor(1);
            std::vector<size_t> weight_tensor_paddings(4);


            weight_tensor_paddings[0] = 0;
            weight_tensor_paddings[1] = 0;
            weight_tensor_paddings[2] = 0;
            weight_tensor_paddings[3] = 0;

            if(weight_tensor->hasAttr("NCE1_Paddings"))
            {
                //Simple rule: greatest padding wins
                std::vector<size_t> existing_weight_tensor_paddings = weight_tensor->get<std::vector<size_t>>("NCE1_Paddings");
                weight_tensor->erase("NCE1_Paddings");
                for(unsigned i = 0; i < existing_weight_tensor_paddings.size();++i)
                    weight_tensor_paddings[i] = std::max(weight_tensor_paddings[i], existing_weight_tensor_paddings[i]);
            }

            weight_tensor->set<std::vector<size_t>>("NCE1_Paddings", weight_tensor_paddings);
        }

        operationIt->set<std::size_t>("NCE1_InputWidthPadded", actual_input_width);
        operationIt->set<std::size_t>("NCE1_OutputWidthPadded", actual_output_width);
    }
}
