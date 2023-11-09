//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "vpux/compiler/dialect/VPUIP/generated/schema/graphfile_generated.h"

#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "ngraph/pass/serialize.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include <vector>

namespace ov::test::subgraph {
typedef std::tuple<std::string> CompressWeightsParameters;

class CompressWeightsTest :
        public testing::WithParamInterface<CompressWeightsParameters>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompressWeightsParameters>& obj) {
        std::string targetDevice;
        std::tie(targetDevice) = obj.param;

        std::ostringstream result;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        const InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP16;
        std::tie(targetDevice) = this->GetParam();

        // NOTE: model is adapted from mobV2_soh test, but scaled up so that the compression is applied
        std::vector<size_t> inputShape = {1, 144, 112, 112};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        // input
        const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        // GroupConv
        const auto groupConvWeights = CommonTestUtils::generate_float_numbers(144 * 3 * 3, -0.2f, 0.2f);
        const auto groupConv =
                ngraph::builder::makeGroupConvolution(params[0], ngPrc, {3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1},
                                                      ngraph::op::PadType::EXPLICIT, 144, 144, false, groupConvWeights);
        // result
        ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(groupConv)};

        function = std::make_shared<ngraph::Function>(results, params, "CompressWeightsTest");
    }
};  // namespace ov::test::subgraph

class CompressWeightsTest_VPU3720 : public CompressWeightsTest, virtual public VpuOv2LayerTest {};

TEST_P(CompressWeightsTest_VPU3720, MLIR_HW) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "CompressWeightsTest only needs to compile the model";
    });
    configuration.emplace(VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND), "NO");
    configuration.emplace(VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS), "compress-weights-btc=true");
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

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

INSTANTIATE_TEST_CASE_P(precommit, CompressWeightsTest_VPU3720,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        CompressWeightsTest::getTestCaseName);
}  // namespace
