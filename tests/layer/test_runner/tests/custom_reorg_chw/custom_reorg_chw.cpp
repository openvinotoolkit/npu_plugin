#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/custom_layer_test.hpp"
#include <iostream>

int main()
{
    const auto is_chw = true;
    const auto order = is_chw ? mv::Order{"NCHW"} : mv::Order{"NHWC"};

    const uint32_t width = 26;
    const uint32_t height = 26;
    const uint32_t channels = 64;
    const uint32_t stride = 2;

    auto test = CustomLayerTest<>("CustomReorgModel");
    test.add_input({width, height, channels, 1}, mv::DType{"Float16"}, order);
    test.add_output({width / stride, height / stride, channels * stride * stride, 1}, mv::DType{"Float16"}, order);

    const auto available_cmx = 60 * 1024;
    uint32_t local_memory_size = 0;
    if (is_chw) {
        test.local_size[0] = [&] {
            const auto memory_demand = width * height * channels / stride * 2 * sizeof(half);
            if (available_cmx > memory_demand) return height;
            if (available_cmx > memory_demand / stride) return height / stride;
            if (available_cmx > memory_demand / height) return 1u;
            throw std::runtime_error("Not enough CMX memory");
        }();
        test.local_size[1] = stride;
        test.local_size[2] = 1;

        test.num_groups[0] = height * channels / stride / stride / test.local_size[0];
        test.num_groups[1] = stride;
        test.num_groups[2] = 1;

        local_memory_size = width * stride * test.local_size[0] * sizeof(half);
    } else {
        const int memory_demand = width * height * 2 * sizeof(half);
        if (memory_demand < available_cmx)
        {
            test.local_size = {height / stride, stride, 1};
            test.num_groups = {channels / stride, stride, 1};
            test.kernel_id = 0;
            local_memory_size = memory_demand / 2;
        }
        else
        {
            test.local_size = {channels / (stride * stride), 1, 1};
            test.num_groups = {stride * stride, 1, 1};
            test.kernel_id = 1;
            local_memory_size = 0;
        }
    }

    test.run(is_chw ? "reorg_chw.elf" : "reorg_hwc.elf",
            {0, 0, width, height, channels, stride, local_memory_size, local_memory_size});
}
