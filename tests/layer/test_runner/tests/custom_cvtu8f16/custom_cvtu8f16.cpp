#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"

int main()
{
    const uint32_t width = 480;
    const uint32_t height = 360;
    const uint32_t channels = 3;
    const uint32_t scale = CustomLayerTest<>::float_as_int(2.0f);
    const uint32_t bias = CustomLayerTest<>::float_as_int(4.0f);

    auto test = CustomLayerTest<>("Convert");
    test.add_input({width, height, channels, 1}, mv::DType{"UInt8"}, mv::Order{"NCHW"});
    test.add_output({width, height, channels, 1}, mv::DType{"Float16"}, mv::Order{"NCHW"});
    test.local_size = {width, 1, 1};
    test.num_groups = {1, height, channels};

    const uint32_t local_src = test.local_size[0] * test.local_size[1] * test.local_size[2] * sizeof(uint8_t);
    const uint32_t local_dst = test.local_size[0] * test.local_size[1] * test.local_size[2] * sizeof(half);

    // KMBQuantizeConversion adds unwanted input dequantize [U8->FP16]
    test.remove_pass("kmb_adapt", "KMBQuantizeConversion");
    test.run("cvtu8f16.elf", {0, 0, scale, bias, local_src, local_dst});

    return 0;
}
