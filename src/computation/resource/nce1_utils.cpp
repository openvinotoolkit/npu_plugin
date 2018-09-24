#include "mcm/computation/resource/nce1_utils.hpp"

mv::ConvolutionParameters mv::fillConvolutionParameters(mv::Data::OpListIterator convIterator)
{
    mv::ConvolutionParameters to_return;
    auto weigth_tensor = convIterator->getInputTensor(1);
    auto input_tensor = convIterator->getInputTensor(0);
    auto output_tensor = convIterator->getOutputTensor(0);

    auto kernel_dimensions = weigth_tensor->getShape();
    auto input_dimensions = input_tensor->getShape();
    auto output_dimensions = output_tensor->getShape();

    to_return.kernel_x = kernel_dimensions[0];
    to_return.kernel_y = kernel_dimensions[1];
    to_return.input_width = input_dimensions[0];
    to_return.input_height = input_dimensions[1];
    to_return.input_channels = input_dimensions[2];
    to_return.output_width = output_dimensions[0];
    to_return.output_height = output_dimensions[1];
    to_return.output_channels = output_dimensions[2];

    if(input_tensor->hasAttr("NCE1_Paddings")) //The input tensor involved in this convolution has already been padded (probably as output tensor of some other convolution)
    {
        std::vector<std::size_t> paddings = input_tensor->get<std::vector<std::size_t>>("NCE1_Paddings");
        to_return.input_width += paddings[0];
        to_return.input_height += paddings[1];
        to_return.input_channels += paddings[2];
    }

    if(output_tensor->hasAttr("NCE1_Paddings"))
    //The output tensor involved in this convolution has already been padded (probably as input tensor of some other convolution)
    //NOTE: Maybe this shouldn't happen at all, but it's better to play safe
    {
        std::vector<std::size_t> paddings = output_tensor->get<std::vector<std::size_t>>("NCE1_Paddings");
        to_return.output_width += paddings[0];
        to_return.output_height += paddings[1];
        to_return.output_channels += paddings[2];
    }

    auto strides = convIterator->get<std::array<unsigned short, 2>>("stride");
    to_return.stride_x = strides[0];
    to_return.stride_y = strides[1];

    auto paddings = convIterator->get<std::array<unsigned short, 4>>("padding");
    to_return.pad_x_up = paddings[0];
    to_return.pad_x_down = paddings[1];
    to_return.pad_y_left = paddings[2];
    to_return.pad_y_right = paddings[3];

    return to_return;
}
