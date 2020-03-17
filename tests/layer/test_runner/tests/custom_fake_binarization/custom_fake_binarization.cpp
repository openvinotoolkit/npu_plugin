#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 56;
    const uint32_t height = 56;
    const uint32_t channels = 256;
    const uint32_t switch_out = 0;
    const uint32_t input_low_high_size = 256;

    auto test = CustomLayerTest<>("FakeBinarization");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_constant("custom_fake_binarization/in1.bin", {1, 1, 256, 1},
        mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_output({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {1, 1, 1};
    test.num_groups = {1, 1, channels};

    test.run("fakebinarization.elf", {0, 1, 0, switch_out, input_low_high_size, width, height});
}