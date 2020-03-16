#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 56;
    const uint32_t height = 56;
    const uint32_t channels = 256;
    const uint32_t levels = 2;
    const uint32_t input_low_size = 256;
    const uint32_t input_high_size = 256;
    const uint32_t output_low_size = 256;
    const uint32_t output_high_size = 256;

    auto test = CustomLayerTest<>("CustomFakeBinarizationModel");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_constant("custom_fake_quantize/in1.bin", {1, 1, 256, 1}, mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_constant("custom_fake_quantize/in2.bin", {1, 1, 256, 1}, mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_constant("custom_fake_quantize/in3.bin", {1, 1, 256, 1}, mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_constant("custom_fake_quantize/in4.bin", {1, 1, 256, 1}, mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_output({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {1, 1, 1};
    test.num_groups = {1, height, 1};

    const uint32_t local_data = width * channels * sizeof(half);

    test.run("fakequantize.elf", {0, 1, 2, 3, 4, 0, levels, input_low_size, input_high_size, output_low_size,
                                  output_high_size, width, channels, local_data, local_data});
}
