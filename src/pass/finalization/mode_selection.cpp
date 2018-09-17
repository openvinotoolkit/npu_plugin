#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/model/types.hpp"

static void modeSelection(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ModeSelection)
        .setFunc(modeSelection)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass selects the appropriate mode for each convolution executable by NCE"
        );
    }
}

mv::ModeSelectionResult optimize_convolution_nce1(mv::Nce1& nce, mv::Data::OpListIterator convIterator)
{
    mv::ModeSelectionNode source;
    auto weigth_tensor = convIterator->getInputTensor(1);
    auto input_tensor = convIterator->getInputTensor(0);
    auto output_tensor = convIterator->getOutputTensor(0);

    auto kernel_dimensions = weigth_tensor->getShape();
    auto input_dimensions = input_tensor->getShape();
    auto output_dimensions = output_tensor->getShape();

    source.parameters.kernel_x = kernel_dimensions[0];
    source.parameters.kernel_y = kernel_dimensions[1];
    source.parameters.input_width = input_dimensions[0];
    source.parameters.input_height = input_dimensions[1];
    source.parameters.input_channels = input_dimensions[2];
    source.parameters.output_width = output_dimensions[0];
    source.parameters.output_height = output_dimensions[1];
    source.parameters.output_channels = output_dimensions[2];

    auto strides = convIterator->getAttr("stride").getContent<mv::UnsignedVector2D>();
    source.parameters.stride_x = strides.e0;
    source.parameters.stride_y = strides.e1;

    source.remaining_output_channels = source.parameters.output_channels;
    return nce.optimize_convolution(source);
}

void write_hardware_attributes(mv::OpModel& om, mv::Data::OpListIterator convIterator, mv::ModeSelectionResult& modes_to_use, mv::Nce1& nce)
{
    // ASSUMPTION: all the splits in modes_to_use are are equal (This assumption is true now, and will be routines to assure it's trueness in the future)

    // Non vector attributes first
    auto input_tensor = convIterator->getInputTensor(0);
    auto input_tensor_dimensions = input_tensor->getShape();

    auto output_tensor = convIterator->getOutputTensor(0);
    auto output_tensor_dimensions = output_tensor->getShape();

    auto weight_tensor = convIterator->getInputTensor(1);
    auto weight_tensor_dimensions = weight_tensor->getShape();

    // Take biggest mode, necessary for getting the actual input channels
    unsigned num_modes_to_use = modes_to_use.distances.size();
    std::vector<unsigned> modes(num_modes_to_use);
    unsigned max_mode = modes_to_use.distances[0].mode;
    for(unsigned i = 0; i < num_modes_to_use; ++i)
    {
        modes[i] = modes_to_use.distances[i].mode;
        if(max_mode < modes[i])
            max_mode = modes[i];
    }

    // Getting real dimensions
    unsigned input_width = input_tensor_dimensions[0];
    unsigned input_height = input_tensor_dimensions[1];
    unsigned input_channels = input_tensor_dimensions[2];
    unsigned output_width = output_tensor_dimensions[0];
    unsigned output_height = output_tensor_dimensions[1];
    unsigned output_channels = output_tensor_dimensions[2];
    unsigned kernel_height = weight_tensor_dimensions[0];

    // Computing actual dimensions
    unsigned actual_input_width = nce.getActualInputWidth(input_width);
    unsigned actual_input_height = input_height;
    unsigned actual_input_channels = nce.getActualInputChannels(input_channels, max_mode);
    unsigned actual_output_width = output_width;
    unsigned actual_output_height = output_height;
    unsigned actual_output_channels = nce.getActualOutputChannels(output_channels);

    // Check if any split is needed
    unsigned actual_total_tensor_size = actual_input_height * actual_input_width * actual_input_channels + actual_output_width * actual_output_height * actual_output_channels;
    unsigned splits_over_height = nce.getSplitsOverH(actual_total_tensor_size);
    unsigned splits_over_input_channels = modes_to_use.distances[0].num_splits; //num_splits must be equal for every mode, see above assumption
    unsigned splitted_input_channels = actual_input_channels / splits_over_input_channels;

    // Handling top output junk and bottom output junk
    if(splits_over_height > 1)
    {
        om.addAttr(convIterator, "NCE1_TopOutputJunk", mv::Attribute(mv::AttrType::IntegerType, 0));
        om.addAttr(convIterator, "NCE1_BottomOutputJunk", mv::Attribute(mv::AttrType::IntegerType, 0));
    }
    else // Should be like this
    {
        om.addAttr(convIterator, "NCE1_TopOutputJunk", mv::Attribute(mv::AttrType::IntegerType, kernel_height-1));
        om.addAttr(convIterator, "NCE1_BottomOutputJunk", mv::Attribute(mv::AttrType::IntegerType, kernel_height-1));
    }

    // Compute paddings for input and output tensors using actual dimensions
    mv::UnsignedVector3D paddingsInputTensor;
    mv::UnsignedVector3D paddingsOutputTensor;

    paddingsInputTensor.e0 = actual_input_width - input_width;
    paddingsInputTensor.e1 = actual_input_height - input_height;
    paddingsInputTensor.e2 = actual_input_channels - input_channels;

    paddingsOutputTensor.e0 = actual_output_width - output_width;
    paddingsOutputTensor.e1 = actual_output_height - output_height;
    paddingsOutputTensor.e2 = actual_output_channels - output_channels;

    input_tensor->addAttr("NCE1_Paddings", mv::Attribute(mv::AttrType::UnsignedVec3DType, paddingsInputTensor));
    output_tensor->addAttr("NCE1_Paddings", mv::Attribute(mv::AttrType::UnsignedVec3DType, paddingsOutputTensor));

    // Compute local line stride
    unsigned local_line_stride = nce.computeLocalLineStride(actual_input_width);
    om.addAttr(convIterator, "NCE1_LocalLineStride", mv::Attribute(mv::AttrType::IntegerType, local_line_stride));

    // Compute DescriptorsSplits
    int descriptor_splits = nce.computeDescriptorSplits(splits_over_height, splits_over_input_channels, actual_output_channels, modes);
    om.addAttr(convIterator, "NCE1_DescriptorSplits", mv::Attribute(mv::AttrType::IntegerType, descriptor_splits));

    // TODO: Streaming mask
    int streaming_mask = 0; // For DDR streaming
    om.addAttr(convIterator, "NCE1_StreamingMask", mv::Attribute(mv::AttrType::IntegerType, streaming_mask));

    // Vector attributes

    std::vector<unsigned> input_channels_per_ram_block(num_modes_to_use);
    std::vector<unsigned> lines_per_channel(num_modes_to_use);
    std::vector<unsigned> local_channel_stride(num_modes_to_use);
    std::vector<unsigned> min_lines(num_modes_to_use);
    for(unsigned i = 0; i < num_modes_to_use; ++i)
    {
        input_channels_per_ram_block[i] = nce.computeInputChannelsPerRamBlock(splitted_input_channels, modes[i]);
        lines_per_channel[i] = nce.computeLinesPerChannel(input_channels_per_ram_block[i], modes[i]);
        local_channel_stride[i] = lines_per_channel[i] * local_line_stride;

        min_lines[i] = 0;
        bool poolEn = false;
        if(poolEn)
            min_lines[i] = 0; //TODO
        else
            min_lines[i] = std::min(kernel_height+1, lines_per_channel[i]);
    }
    om.addAttr(convIterator, "NCE1_InputChannelsRamBlock", mv::Attribute(mv::AttrType::FloatVecType, local_line_stride));
    om.addAttr(convIterator, "NCE1_LinesPerChannel", mv::Attribute(mv::AttrType::FloatVecType, lines_per_channel));
    om.addAttr(convIterator, "NCE1_LocalChannelStride", mv::Attribute(mv::AttrType::FloatVecType, local_channel_stride));
    om.addAttr(convIterator, "NCE1_MinLines", mv::Attribute(mv::AttrType::FloatVecType, min_lines));
}

void modeSelection(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::Nce1 nce;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        switch(opIterator->getOpType())
        {
            case mv::OpType::Conv2D:
                mv::ModeSelectionResult modes = optimize_convolution_nce1(nce, opIterator);
                write_hardware_attributes(om, opIterator, modes, nce);
                break;
            /*
            case mv::OpType::FullyConnected:
                //TODO: Write mode 0 or 4
                break;
            case mv::OpType::MaxPool2D:
            case mv::OpType::AvgPool2D:
                //TODO: Write mode 0 or 4
                break;
            */
        }
    }
}
