#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 26;
    const uint32_t height = 26;
    const float factor = 2.0f;
    const uint32_t out_width = 52;
    const uint32_t out_height = 52;
    const uint32_t channels = 128;

    auto test = CustomLayerTest<>("CustomResampleAAModel");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_output({out_width, out_height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {1, 1, channels};
    test.num_groups = {1, out_height, 1};

    const int r = (factor > 1.0f) ? 2 : (int)ceil(1.0f / factor);
    const uint32_t local_src = width * (2 * r + (int)std::ceil(test.local_size[1] / factor))
            * test.local_size[2] * sizeof(half);
    const uint32_t local_dst = out_width * test.local_size[1] * test.local_size[2] * sizeof(half);
    test.run("resample_AA.elf", {0, 0, local_src, local_dst, width, height,
            CustomLayerTest<>::float_as_int(2.0f), out_width, out_height, channels});
}