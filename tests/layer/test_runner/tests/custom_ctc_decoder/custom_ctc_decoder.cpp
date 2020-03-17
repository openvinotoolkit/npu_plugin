#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 71;
    const uint32_t height = 1;
    const uint32_t channels = 88;

    auto test = CustomLayerTest<>("CTCDecoder");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_constant("custom_ctc_decoder/in1.bin", {channels, 1, 1, 1}, mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_output({channels, 1, 1, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {1, 1, 1};
    test.num_groups = {1, 1, 1};

    const uint32_t local_data = width * height * channels * sizeof(half);

    test.run("ctc.elf", {0, 1, 0, width, height, channels, local_data, local_data});
}