#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t top_width = 64;
    const uint32_t top_height = 48;
    const uint32_t width = 64;
    const uint32_t height = 48;
    const uint32_t channels = 64;
    const uint32_t displacement = 8;
    const uint32_t pad = 8;
    const uint32_t neighborhood_grid_radius = 4;
    const uint32_t neighborhood_grid_width = 9;
    const uint32_t kernel_size = 1;
    const uint32_t stride1 = 1;
    const uint32_t stride2 = 2;

    const uint32_t top_channels = 81;

    auto test = CustomLayerTest<>("Correlate");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_constant("custom_correlate/in1.bin", {width, height, channels, 1},
        mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_output({top_width, top_height, top_channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {1, 1, 1};
    test.num_groups = {top_height, 1, 1};

    test.run("correlate.elf", {0, 1, 0, top_width, top_height, width, height, channels, displacement, pad,
            neighborhood_grid_radius, neighborhood_grid_width, kernel_size, stride1, stride2});
}
