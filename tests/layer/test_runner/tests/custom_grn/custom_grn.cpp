#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 224;
    const uint32_t height = 128;
    const uint32_t channels = 24;
    const uint32_t bias = CustomLayerTest<>::float_as_int(1.0f);

    auto test = CustomLayerTest<>("GRN");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_output({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {width, 1, 1};
    test.num_groups = {1, height, 1};

    const uint32_t local_data = test.local_size[0] * test.local_size[1] * test.local_size[2]
            * channels * sizeof(half);

    test.run("grn.elf", {0, 0, local_data, local_data, channels, bias});
}