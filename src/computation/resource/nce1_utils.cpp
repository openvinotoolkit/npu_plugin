#include "mcm/computation/resource/nce1_utils.hpp"

mv::ConvolutionParameters mv::fillKernel2DOperationParameters(mv::Data::OpListIterator opIterator, bool add_padding)
{
    mv::ConvolutionParameters to_return;
    auto input_tensor = opIterator->getInputTensor(0);
    auto output_tensor = opIterator->getOutputTensor(0);

    auto input_dimensions = input_tensor->getShape();
    auto output_dimensions = output_tensor->getShape();

    if(opIterator->getOpType() == "Conv")
    {
        auto weigth_tensor = opIterator->getInputTensor(1);
        auto kernel_dimensions = weigth_tensor->getShape();
        to_return.kernel_width = kernel_dimensions[0];
        to_return.kernel_height = kernel_dimensions[1];
    }
    else if(opIterator->getOpType() == "AveragePool" || opIterator->getOpType() == "MaxPool")
    {
        auto kernel_dimensions = opIterator->get<std::array<short unsigned, 2>>("kSize");
        to_return.kernel_width = kernel_dimensions[0];
        to_return.kernel_height = kernel_dimensions[1];
    }

    to_return.input_width = input_dimensions[0];
    to_return.input_height = input_dimensions[1];
    to_return.input_channels = input_dimensions[2];
    to_return.output_width = output_dimensions[0];
    to_return.output_height = output_dimensions[1];
    to_return.output_channels = output_dimensions[2];

    if(add_padding)
    {
        std::vector<size_t> existing_output_tensor_paddings = output_tensor->get<std::vector<size_t>>("NCE1_Paddings");
        std::vector<size_t> existing_input_tensor_paddings = input_tensor->get<std::vector<size_t>>("NCE1_Paddings");

        to_return.input_width += existing_input_tensor_paddings[0];
        to_return.input_height += existing_input_tensor_paddings[1];
        to_return.input_channels += existing_input_tensor_paddings[2];
        to_return.output_width += existing_output_tensor_paddings[0];
        to_return.output_height += existing_output_tensor_paddings[1];
        to_return.output_channels += existing_output_tensor_paddings[2];
    }

    auto strides = opIterator->get<std::array<unsigned short, 2>>("stride");
    to_return.stride_vertical = strides[0];
    to_return.stride_horizontal = strides[1];

    auto paddings = opIterator->get<std::array<unsigned short, 4>>("padding");
    to_return.pad_x_up = paddings[0];
    to_return.pad_x_down = paddings[1];
    to_return.pad_y_left = paddings[2];
    to_return.pad_y_right = paddings[3];

    return to_return;
}
