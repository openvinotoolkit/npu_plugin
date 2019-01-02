#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/resource/nce1_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void optimizePoolingsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(OptimizePoolings)
        .setFunc(optimizePoolingsFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass optimizes each pooling layer for the NCE"
        );
    }
}


bool write_hardware_attributes_pooling(mv::OpModel& om, mv::Data::OpListIterator poolIterator, mv::ModeSelectionResult& modes_to_use, mv::Nce1& nce)
{
    // ASSUMPTION: all the splits in modes_to_use are are equal (This assumption is true now, and there will be routines to assure it's trueness in the future)

    // Scalar attributes first
    auto input_tensor = poolIterator->getInputTensor(0);
    auto input_tensor_dimensions = input_tensor->getShape();
    auto output_tensor = poolIterator->getOutputTensor(0);
    auto output_tensor_dimensions = output_tensor->getShape();


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

    std::size_t input_channels = input_tensor_dimensions[2];
    std::size_t input_width = input_tensor_dimensions[0];

    //ASSUMPTION: Mode selection always takes place after MX paddings pass.
    //CONSEQUENCE: There is no need to check for NCE1_Paddings, as each tensor involved in HW convolution
    //will have it

    //Prepass takes care of padding of width for each input/output tensor to HW operation.
    //Prepass also takes care of padding of channels for output tensor of an HW operation.
    //Prepass also takes care of padding of output channels for weight tensor.

    std::vector<std::size_t> input_tensor_paddings = input_tensor->get<std::vector<std::size_t>>("NCE1_Paddings");
    input_width += input_tensor_paddings[0];
    input_channels += input_tensor_paddings[2];
    input_tensor->erase("NCE1_Paddings");

    std::vector<std::size_t> output_tensor_paddings = output_tensor->get<std::vector<std::size_t>>("NCE1_Paddings");
    output_tensor->erase("NCE1_Paddings");

    //We must take care of any additional padding that might be needed for input channels due to the maximum mode selected
    //For both input tensor and weight tensor
    auto input_channels_needed_by_max_selected_mode = nce.computeActualInputChannels(input_channels, mv::Mode4);
    input_tensor_paddings[2] += input_channels_needed_by_max_selected_mode - input_channels;

    //Also output channels paddings need to be updated, because they could have been changed
    //(e.g. another convolution using the same tensor as input and padding the channels)
    //They need to be updated just in weight tensor
    output_tensor_paddings[2] += input_channels_needed_by_max_selected_mode - input_channels;

    input_channels = input_channels_needed_by_max_selected_mode;


    std::size_t kernel_height = poolIterator->get<std::array<short unsigned, 2>>("kSize")[1];

    // Compute local line stride
    unsigned local_line_stride = nce.computeLocalLineStride(input_width);
    poolIterator->set<std::size_t>("NCE1_LocalLineStride", (std::size_t)local_line_stride);

    // TODO: Streaming mask
    unsigned streaming_mask = 1; // For DDR streaming
    poolIterator->set<std::size_t>("NCE1_StreamingMask", (std::size_t)streaming_mask);

    // Max performed output channels
    poolIterator->set<std::size_t>("NCE1_MaxOutputChannelsPerformed", (std::size_t)max_output_channels_performed);

    // -------------------VECTOR ATTRIBUTES----------------
    std::vector<std::size_t> input_channels_per_ram_block(num_modes_to_use);
    std::vector<std::size_t> lines_per_channel(num_modes_to_use);
    std::vector<std::size_t> local_channel_stride(num_modes_to_use);
    std::vector<std::size_t> min_lines(num_modes_to_use);
    for(unsigned i = 0; i < num_modes_to_use; ++i)
    {
        input_channels_per_ram_block[i] = 1;
        lines_per_channel[i] = nce.computeLinesPerChannel(input_channels, local_line_stride, modes[i]);
        local_channel_stride[i] = lines_per_channel[i] * local_line_stride;

        //std::cout << "input_channels_per_ram_block[i] " << input_channels_per_ram_block[i] << std::endl;
        //std::cout << "lines_per_channel[i] " << lines_per_channel[i] << std::endl;
        //std::cout << "local_channel_stride[i] " << lines_per_channel[i] << std::endl;

        min_lines[i] = kernel_height * 2;
    }
    poolIterator->set<std::vector<std::size_t>>("NCE1_Modes", modes);
    poolIterator->set<std::vector<std::size_t>>("NCE1_OutputChannelsPerformed", output_channels_performed);
    poolIterator->set<std::vector<std::size_t>>("NCE1_InputChannelsRamBlock", input_channels_per_ram_block);
    poolIterator->set<std::vector<std::size_t>>("NCE1_LinesPerChannel", lines_per_channel);
    poolIterator->set<std::vector<std::size_t>>("NCE1_LocalChannelStride", local_channel_stride);
    poolIterator->set<std::vector<std::size_t>>("NCE1_MinLines", min_lines);

    //Input Tensor and weigth paddings need to be rewritten, as they may have changed due to mode selected
    input_tensor->set<std::vector<std::size_t>>("NCE1_Paddings", input_tensor_paddings);
    output_tensor->set<std::vector<std::size_t>>("NCE1_Paddings", output_tensor_paddings);

    poolIterator->set<std::size_t>("NCE1_InputChannelsPadded", input_channels);
    poolIterator->set<std::size_t>("NCE1_OutputChannelsPadded", input_channels);

    //Marking the convolution as optimized
    poolIterator->set<bool>("NCE1_Optimized", true);

    //This check has a sense only for cascades of poolings since HW convolutions still have to be optimized
    auto parent_op = om.getSourceOp(input_tensor);
    if(parent_op->hasAttr("NCE1_Optimized") && (parent_op->getOpType() == "AvgPool" || parent_op->getOpType() == "MaxPool"))
        return true;
    else
        return false;
}

mv::ModeSelectionResult optimize_pooling_nce1(mv::Nce1& nce, mv::Data::OpListIterator poolIterator, mv::OpModel&)
{
    mv::ModeSelectionNode source;
    source.parameters = mv::fillKernel2DOperationParameters(poolIterator, true);

    source.remaining_output_channels = source.parameters.output_channels;
    return nce.optimize_pooling(source);
}

void optimizePoolingsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    std::cout << "HW pooling optimization pass started" << std::endl;
    mv::OpModel om(model);
    mv::Nce1 nce;

    //Maybe an algorithm for topological sorting is needed, but for now a stack instead of queue will work
    std::stack<mv::Data::OpListIterator> to_be_optimized;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (opIterator->getOpType() == "MaxPool" || opIterator->getOpType() == "AvgPool")
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
        auto modes = optimize_pooling_nce1(nce, opIterator, om);
        if(write_hardware_attributes_pooling(om, opIterator, modes, nce))
            std::cout << "FOOOOOOOOOOOOOOOL!!! Reoptimization is needed (pooling)";

    }

    std::cout << "HW pooling optimization pass ended" << std::endl;

}
