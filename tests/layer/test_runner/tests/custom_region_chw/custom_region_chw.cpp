#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"
#include <iostream>


int main()
{
    const auto is_chw = true;
    const auto order = is_chw ? mv::Order{"NCHW"} : mv::Order{"NHWC"};

    const uint32_t width = 13;
    const uint32_t height = 13;
    const uint32_t channels = 125;
    const uint32_t classes = 20;
    const uint32_t coords  = 4;
    const uint32_t num     = 5;

    auto test = CustomLayerTest<>("RegionYolo");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, order);
    test.add_output({width, height, channels, 1}, mv::DType{"Float16"}, order);
    test.local_size = {(width + 7)/8*8, 1, 1};
    test.num_groups = {height, num, 1};
    test.kernel_id = is_chw ? 0 : 1;

    const uint32_t local_data = (width + 7)/8*8 * (classes + coords + 1) * sizeof(half);

    test.run("region.elf", {0, 0, local_data, local_data, width, height, classes, coords, num});

    return 0;
}
