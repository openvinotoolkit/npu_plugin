#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/model/types.hpp"

static void myriadXPaddings(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

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


void myriadXPaddings(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Nce1 nce;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        if(operationIt->getOpType() != mv::OpType::Conv2D)
            continue;
        auto input_tensor = operationIt->getInputTensor(0);
        auto input_tensor_dimension = input_tensor->getShape();

        auto output_tensor = operationIt->getOutputTensor(0);
        auto output_tensor_dimension = output_tensor->getShape();

        size_t input_width = input_tensor_dimension[0];
        size_t input_height = input_tensor_dimension[1];
        size_t input_channels = input_tensor_dimension[2];

        size_t actual_input_width = nce.getActualInputWidth(input_width);
        size_t actual_input_height = nce.getActualInputHeight(input_height);
        size_t actual_input_channels = nce.getActualInputChannels(input_channels);

        size_t output_width = output_tensor_dimension[0];
        size_t output_height = output_tensor_dimension[1];
        size_t output_channels = output_tensor_dimension[2];

        size_t actual_output_width = nce.getActualOutputWidth(output_width);
        size_t actual_output_height = nce.getActualOutputHeight(output_height);
        size_t actual_output_channels = nce.getActualOutputChannels(output_channels);

        mv::dynamic_vector<size_t> input_tensor_paddings(input_tensor_dimension.ndims());
        mv::dynamic_vector<size_t> output_tensor_paddings(output_tensor_dimension.ndims());

        input_tensor_paddings[0] = actual_input_width - input_width;
        input_tensor_paddings[1] = actual_input_height - input_height;
        input_tensor_paddings[2] = actual_input_channels - input_channels;

        output_tensor_paddings[0] = actual_output_width - output_width;
        output_tensor_paddings[1] = actual_output_height - output_height;
        output_tensor_paddings[2] = actual_output_channels - output_channels;

        if(input_tensor->hasAttr("NCE1_Paddings"))
        {
            //Simple rule: greater padding wins
            mv::dynamic_vector<size_t> existing_input_tensor_paddings = input_tensor->getAttr("NCE1_Paddings").getContent<mv::dynamic_vector<size_t>>();
            input_tensor->removeAttr("NCE1_Paddings");
            for(unsigned i = 0; i < existing_input_tensor_paddings.size();++i)
                input_tensor_paddings[i] = std::max(input_tensor_paddings[i], existing_input_tensor_paddings[i]);
        }

        if(output_tensor->hasAttr("NCE1_Paddings"))
        {
            //Simple rule: greater padding wins
            mv::dynamic_vector<size_t> existing_output_tensor_paddings = input_tensor->getAttr("NCE1_Paddings").getContent<mv::dynamic_vector<size_t>>();
            output_tensor->removeAttr("NCE1_Paddings");
            for(unsigned i = 0; i < existing_output_tensor_paddings.size();++i)
                output_tensor_paddings[i] = std::max(output_tensor_paddings[i], existing_output_tensor_paddings[i]);
        }
        dm.addAttr(input_tensor, "NCE1_Paddings", mv::Attribute(mv::AttrType::UnsignedVecType, input_tensor_paddings));
        dm.addAttr(output_tensor, "NCE1_Paddings", mv::Attribute(mv::AttrType::UnsignedVecType, output_tensor_paddings));
    }
}
