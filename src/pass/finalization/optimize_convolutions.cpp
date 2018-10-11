#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/resource/nce1_utils.hpp"

static void optimizeConvolutions(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(OptimizeConvolutions)
        .setFunc(optimizeConvolutions)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass selects the appropriate mode for each convolution executable by NCE"
        );
    }
}


bool write_hardware_attributes(mv::OpModel& om, mv::Data::OpListIterator convIterator, mv::ModeSelectionResult& modes_to_use, mv::Nce1& nce)
{
    // ASSUMPTION: all the splits in modes_to_use are are equal (This assumption is true now, and there will be routines to assure it's trueness in the future)

    // Scalar attributes first
    auto input_tensor = convIterator->getInputTensor(0);
    auto input_tensor_dimensions = input_tensor->getShape();

    auto output_tensor = convIterator->getOutputTensor(0);
    auto output_tensor_dimensions = output_tensor->getShape();

    auto weight_tensor = convIterator->getInputTensor(1);
    auto weight_tensor_dimensions = weight_tensor->getShape();

    // Take biggest mode, necessary for getting the actual input channels
    unsigned num_modes_to_use = modes_to_use.distances.size();
    std::vector<size_t> modes(num_modes_to_use);
    std::vector<size_t> output_channels_performed(num_modes_to_use);

    unsigned max_mode = modes_to_use.distances[0].mode;
    unsigned max_output_channels_performed = modes_to_use.distances[0].performed_output_channels;
    for(unsigned i = 0; i < num_modes_to_use; ++i)
    {
        modes[i] = modes_to_use.distances[i].mode;
        if(max_mode < modes[i])
            max_mode = modes[i];
        output_channels_performed[i] = modes_to_use.distances[i].performed_output_channels;
        if(max_output_channels_performed < output_channels_performed[i])
            max_output_channels_performed = output_channels_performed[i];
    }

    // Getting real dimensions
    std::size_t input_width = input_tensor_dimensions[0];
    std::size_t input_height = input_tensor_dimensions[1];
    std::size_t input_channels = input_tensor_dimensions[2];

    std::size_t output_channels = output_tensor_dimensions[2];

    std::size_t kernel_height = weight_tensor_dimensions[1];

    //ASSUMPTION: Mode selection always takes place after MX paddings pass.
    //CONSEQUENCE: There is no need to check for NCE1_Paddings, as each tensor involved in HW convolution
    //will have it

    //Prepass takes care of padding of width for each input/output tensor to HW operation.
    //Prepass also takes care of padding of channels for output tensor of an HW operation.
    //Prepass also takes care of padding of output channels for weight tensor.

    std::vector<std::size_t> input_tensor_paddings = input_tensor->get<std::vector<std::size_t>>("NCE1_Paddings");
    input_width += input_tensor_paddings[0];
    input_height += input_tensor_paddings[1];
    input_channels += input_tensor_paddings[2];
    input_tensor->erase("NCE1_Paddings");

    auto weight_tensor_paddings = weight_tensor->get<std::vector<std::size_t>>("NCE1_Paddings");
    weight_tensor->erase("NCE1_Paddings");

    //We must take care of any additional padding that might be needed for input channels due to the maximum mode selected
    //For both input tensor and weight tensor
    auto input_channels_needed_by_max_selected_mode = nce.computeActualInputChannels(input_channels, max_mode);
    input_tensor_paddings[2] += input_channels_needed_by_max_selected_mode - input_channels;
    weight_tensor_paddings[2] +=  input_tensor_paddings[2];
    input_channels = input_channels_needed_by_max_selected_mode;

    //Also output channels paddings need to be updated, because they could have been changed
    //(e.g. another convolution using the same tensor as input and padding the channels)
    //They need to be updated just in weight tensor
    auto real_output_channels = modes_to_use.nodes[0].remaining_output_channels;
    weight_tensor_paddings[3] += real_output_channels - output_channels;

    // Check if any split over input channel is needed
    unsigned splits_over_input_channels = modes_to_use.distances[0].num_splits; //num_splits MUST be equal for every mode, see above assumption

    unsigned splitted_input_channels = input_channels / splits_over_input_channels;
    convIterator->set<std::size_t>("NCE1_SplitsOverInputChannels", (std::size_t)splits_over_input_channels);

    // Compute local line stride
    unsigned local_line_stride = nce.computeLocalLineStride(input_width);
    convIterator->set<std::size_t>("NCE1_LocalLineStride", (std::size_t)local_line_stride);

    // TODO: Streaming mask
    unsigned streaming_mask = 0; // For DDR streaming
    convIterator->set<std::size_t>("NCE1_StreamingMask", (std::size_t)streaming_mask);

    // Max performed output channels
    convIterator->set<std::size_t>("NCE1_MaxOutputChannelsPerformed", (std::size_t)max_output_channels_performed);

    // -------------------VECTOR ATTRIBUTES----------------
    std::vector<std::size_t> input_channels_per_ram_block(num_modes_to_use);
    std::vector<std::size_t> lines_per_channel(num_modes_to_use);
    std::vector<std::size_t> local_channel_stride(num_modes_to_use);
    std::vector<std::size_t> min_lines(num_modes_to_use);
    for(unsigned i = 0; i < num_modes_to_use; ++i)
    {
        input_channels_per_ram_block[i] = nce.computeInputChannelsPerRamBlock(splitted_input_channels, modes[i]);
        lines_per_channel[i] = nce.computeLinesPerChannel(splitted_input_channels, local_line_stride, modes[i]);
        local_channel_stride[i] = lines_per_channel[i] * local_line_stride;

        //std::cout << "input_channels_per_ram_block[i] " << input_channels_per_ram_block[i] << std::endl;
        //std::cout << "lines_per_channel[i] " << lines_per_channel[i] << std::endl;
        //std::cout << "local_channel_stride[i] " << lines_per_channel[i] << std::endl;

        min_lines[i] = 0;
        bool poolEn = false;
        if(poolEn)
            min_lines[i] = 0; //TODO
        else
            min_lines[i] = std::min(kernel_height + 1, lines_per_channel[i]);
    }
    convIterator->set<std::vector<std::size_t>>("NCE1_Modes", modes);
    convIterator->set<std::vector<std::size_t>>("NCE1_OutputChannelsPerformed", output_channels_performed);
    convIterator->set<std::vector<std::size_t>>("NCE1_InputChannelsRamBlock", input_channels_per_ram_block);
    convIterator->set<std::vector<std::size_t>>("NCE1_LinesPerChannel", lines_per_channel);
    convIterator->set<std::vector<std::size_t>>("NCE1_LocalChannelStride", local_channel_stride);
    convIterator->set<std::vector<std::size_t>>("NCE1_MinLines", min_lines);
    convIterator->set<std::size_t>("NCE1_InputChannelsPadded", splitted_input_channels);

    //Input Tensor and weigth paddings need to be rewritten, as they may have changed due to mode selected
    input_tensor->set<std::vector<std::size_t>>("NCE1_Paddings", input_tensor_paddings);
    weight_tensor->set<std::vector<std::size_t>>("NCE1_Paddings", weight_tensor_paddings);

    //Marking the convolution as optimized
    convIterator->set<bool>("NCE1_Optimized", true);

    if(om.getSourceOp(input_tensor)->hasAttr("NCE1_Optimized"))
        return true;
    else
        return false;
}

mv::ModeSelectionResult optimize_convolution_nce1(mv::Nce1& nce, mv::Data::OpListIterator convIterator, mv::OpModel& om)
{
    mv::ModeSelectionNode source;
    source.parameters = mv::fillKernel2DOperationParameters(convIterator, true);

    source.remaining_output_channels = source.parameters.output_channels;
    return nce.optimize_convolution(source);
}

void optimizeConvolutions(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    std::cout << "HW optimization convolution pass started" << std::endl;
    mv::OpModel om(model);
    mv::Nce1 nce;

    //Maybe an algorithm for topological sorting is needed, but for now a stack instead of queue will work
    std::stack<mv::Data::OpListIterator> to_be_optimized;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (opIterator->getOpType() == mv::OpType::Conv2D)
        {

            if(!opIterator->hasAttr("NCE1_Compatible"))
                continue;
            if(!opIterator->get<int>("NCE1_Compatible"))
                continue;
            to_be_optimized.push(opIterator);

        }

    }

    while(to_be_optimized.size() > 0)
    {

        auto opIterator = to_be_optimized.top();
        std::cout << "Optimizing " << opIterator->getName() << std::endl;
        to_be_optimized.pop();
        auto modes = optimize_convolution_nce1(nce, opIterator, om);
        if(write_hardware_attributes(om, opIterator, modes, nce))
            std::cout << "FOOOOOOOOOOOOOOOOOOOL! Reoptimization is needed! :(" << std::endl;

    }

    std::cout << "HW optimization convolution pass ended" << std::endl;

}
