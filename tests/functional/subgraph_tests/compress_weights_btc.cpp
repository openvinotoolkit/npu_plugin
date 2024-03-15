//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "vpux/compiler/dialect/VPUIP/generated/schema/graphfile_generated.h"

#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "ngraph/pass/serialize.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include <vector>
using namespace ov::test::utils;

namespace ov::test {

typedef std::tuple<ov::element::Type> CompressWeightsParameters;

class CompressWeightsTest :
        public testing::WithParamInterface<CompressWeightsParameters>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompressWeightsParameters>& obj) {
        ov::element::Type inType = std::get<0>(obj.param);

        std::ostringstream result;
        result << "InputPrec=" << inType;
        return result.str();
    }

protected:
    void SetUp() override {
        inType = std::get<0>(GetParam());

        // NOTE: model is adapted from mobV2_soh test, but scaled up so that the compression is applied
        std::vector<size_t> inputShape = {1, 144, 112, 112};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        // input
        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};
        // GroupConv
        const auto groupConvWeights = generate_float_numbers(144 * 3 * 3, -0.2f, 0.2f);
        const auto groupConv =
                ngraph::builder::makeGroupConvolution(params[0], inType, {3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1},
                                                      ov::op::PadType::EXPLICIT, 144, 144, false, groupConvWeights);
        // result
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(groupConv)};

        function = std::make_shared<ov::Model>(results, params, "CompressWeightsTest");
    }
};  // namespace ov::test::subgraph

class CompressWeightsTest_NPU3720 : public CompressWeightsTest, public VpuOv2LayerTest {};

TEST_P(CompressWeightsTest_NPU3720, HW) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "CompressWeightsTest only needs to compile the model";
    });
    configuration.emplace(ov::intel_vpux::use_elf_compiler_backend.name(), "NO");
    configuration.emplace(ov::intel_vpux::compilation_mode_params.name(), "compress-weights-btc=true");
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);

    // save the blob into a stringstream
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    compiledModel.export_model(ss);

    // read blob from stringstream with MVCNN
    ss.seekg(0, std::ios::end);
    size_t dataSize = ss.tellg();
    ss.seekg(0, std::ios::beg);

    std::vector<uint8_t> blobBin(dataSize);
    ss.read(reinterpret_cast<char*>(blobBin.data()), dataSize);
    const MVCNN::GraphFile* graphFile = MVCNN::GetGraphFile(blobBin.data());

    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dmaTaskList = nullptr;
    auto taskLists = graphFile->task_lists();
    VPUX_THROW_UNLESS(taskLists, "Blob contains no taskLists");
    for (const auto& taskListItem : *taskLists) {
        const auto content = taskListItem->content();
        if (content->size() == 0) {
            continue;
        }
        const auto task0_type = content->Get(0)->task_type();
        if (task0_type == MVCNN::SpecificTask_NNDMATask) {
            dmaTaskList = taskListItem->content();
        }
    }
    VPUX_THROW_UNLESS(dmaTaskList != nullptr, "Blob contains no DMA tasks");

    bool hasCompressedDMATasks = false;
    for (unsigned dmaTaskListId = 0; dmaTaskListId < (*dmaTaskList).size(); dmaTaskListId++) {
        auto task = (*dmaTaskList)[dmaTaskListId];
        const MVCNN::NNDMATask* dmaTask = task->task_as_NNDMATask();

        if (dmaTask->compression()) {
            hasCompressedDMATasks = true;

            auto srcDims = dmaTask->src()->dimensions();
            auto dstDims = dmaTask->dst()->dimensions();

            VPUX_THROW_UNLESS(
                    srcDims->Get(0) < dstDims->Get(0),
                    "DecompressDMA src()->dimensions()->Get(0) should be less than dst()->dimensions()->Get(0)");

            dmaTask->src()->data();
        }
    }
    VPUX_THROW_UNLESS(hasCompressedDMATasks, "Blob contains no compressed DMA tasks");
}

INSTANTIATE_TEST_CASE_P(precommit, CompressWeightsTest_NPU3720, ::testing::Combine(::testing::Values(ov::element::f16)),
                        CompressWeightsTest::getTestCaseName);
}  // namespace ov::test
