#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 188;
    const uint32_t height = 96;
    const uint32_t channels = 3;

    auto test = CustomLayerTest<>("SpatialTransform");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.add_constant("custom_st/in1.bin", {3, 2, 1, 1}, mv::DType{"Float16"}, mv::Order{"NHWC"});
    test.add_output({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {1, 1, 1};
    test.num_groups = {1, height, 1};

    const uint32_t max_width = 512;  // kernel const
    const uint32_t local_data = std::min(max_width, width) * test.local_size[1] * channels * sizeof(half);

    test.run("st.elf", {0, 1, 0, channels, width, local_data});
}