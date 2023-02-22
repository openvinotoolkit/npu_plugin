//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/parameter.hpp>
#include <shared_test_classes/single_layer/mvn.hpp>
#include <single_layer_tests/mvn.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/op/mvn.hpp"
#include "single_layer_tests/mvn.hpp"

namespace LayerTestsDefinitions {

class KmbMvnLayerTest : public Mvn1LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvnLayerTest, basicTest) {
    Run();
}

class KmbMvnLayerTestMLIR : public Mvn1LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvnLayerTestMLIR, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

// order permutations in test representable notation
enum class HWLayout : uint8_t {
    ANY = 0,
    NCHW = 1,
    NHWC = 2,
    NCWH = 201,
    NWHC = 202,
};

typedef std::tuple<InferenceEngine::SizeVector,  // Input shapes
                   InferenceEngine::Precision,   // Input precision
                   HWLayout,                     // Input layout
                   ngraph::AxisSet,              // Reduction axes
                   bool,                         // Across channels
                   bool,                         // Normalize variance
                   double,                       // Epsilon
                   std::string                   // Device name
                   >
        mvn2Params;

/**
 * For testing particular kernel with variable order currently there is no way to generalize that in KMBLayerTestCommon
 */
class KmbMvnLayerTestMLIR_VPU3720 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<mvn2Params> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
    }
    void SetUp() override {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        ngraph::AxisSet axes;
        bool acrossChanels, normalizeVariance;
        double eps;
        HWLayout hwl;

        std::tie(inputShapes, inputPrecision, hwl, axes, acrossChanels, normalizeVariance, eps, targetDevice) =
                this->GetParam();

        auto make_transpose_order = [hwl]() -> std::vector<int64_t> {
            switch (hwl) {
            case HWLayout::NHWC:
                return {0, 2, 3, 1};
            case HWLayout::NWHC:
                return {0, 3, 2, 1};
            case HWLayout::NCWH:
                return {0, 1, 3, 2};
            default:
                return {0, 1, 2, 3};
            };
        };

        const auto param = ngraph::builder::makeParams(ngraph::element::f16, {inputShapes});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));

        // order handling
        auto input_transpose_order =
                ngraph::opset8::Constant::create(ngraph::element::i64, {4}, make_transpose_order());
        auto input_transpose = std::make_shared<ngraph::opset1::Transpose>(paramOuts[0], input_transpose_order);

        auto make_mvn_kernel = [&]() {
            if (!axes.empty()) {
                return std::dynamic_pointer_cast<ngraph::op::MVN>(
                        ngraph::builder::makeMVN(input_transpose, axes, normalizeVariance, eps));
            }
            return std::dynamic_pointer_cast<ngraph::op::MVN>(
                    ngraph::builder::makeMVN(input_transpose, acrossChanels, normalizeVariance, eps));
        };
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(make_mvn_kernel())};
        function = std::make_shared<ngraph::Function>(results, param, "MVN1");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvn2Params>& obj) {
        mvn1Params mvn1;
        HWLayout hwl;

        std::tie(std::get<0>(mvn1), std::get<1>(mvn1), hwl, std::get<2>(mvn1), std::get<3>(mvn1), std::get<4>(mvn1),
                 std::get<5>(mvn1), std::get<6>(mvn1)) = obj.param;
        auto layout_name = [hwl]() -> std::string {
            switch (hwl) {
            case HWLayout::NHWC:
                return "NHWC";
            case HWLayout::NWHC:
                return "NWHC";
            case HWLayout::NCWH:
                return "NCWH";
            default:
                return "NCHW";
            };
        };

        std::ostringstream resultName;
        resultName << Mvn1LayerTest::getTestCaseName(testing::TestParamInfo<mvn1Params>(mvn1, 0)) << "_";
        resultName << "Layout=" << layout_name();
        return resultName.str();
    }
};

TEST_P(KmbMvnLayerTestMLIR_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

class KmbMvn6LayerTest : public Mvn6LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvn6LayerTest, basicTest) {
    Run();
}

class KmbMvn6LayerTestMLIR : public Mvn6LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvn6LayerTestMLIR, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

// Accuracy problem for 3-dim input tensor when acrossChannels = false
// Traced in Jira S#48579
const std::vector<std::vector<size_t>> inputShapes3D = {{1, 32, 17}, {1, 37, 9}};

const std::vector<ngraph::AxisSet> emptyReductionAxes = {{}};

const std::vector<bool> acrossChannels3D = {true};

const std::vector<std::vector<size_t>> inputShapes = {
        {1, 16, 5, 8},

#if 0
// Test fails for accuracy when Number of channels > 1 (tracked in Jira S#48579)
    {3, 32, 17},
    {3, 37, 9},
// Batch size > 1 is not supported by Soft and Custom Layer MVN implementation
    {2, 19, 5, 10},
    {7, 32, 2, 8},
    {5, 8, 3, 5},
    {4, 41, 6, 9},
// Currently input dim > 4 is not supported by VPUx-plugin and mcmCompiler
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
#endif
};

// only 4d and 5d input shape is supported according to the OpenVino documentation
const std::vector<std::vector<size_t>> MLIRinputShapes = {
        {1, 16, 5, 8},
#if 0
// Batch size > 1 is not supported by Soft and Custom Layer MVN implementation
    {2, 19, 5, 10},
    {7, 32, 2, 8},
    {5, 8, 3, 5},
    {4, 41, 6, 9},
// Currently input dim > 4 is not supported by VPUx-plugin
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
#endif
};
const std::vector<std::vector<size_t>> MLIRinputShapesForOrder = {{1, 4, 2, 1024}};

const std::vector<std::vector<size_t>> inputShapes3720 = {
        {5, 2, 7, 3},   {1, 3, 17, 21}, {1, 4, 512, 1}, {2, 5, 13, 27}, {1, 7, 55, 33},
        {5, 8, 3, 5},   {4, 9, 7, 2},   {1, 16, 5, 8},  {3, 13, 9, 9},  {1, 16, 12, 11},
        {2, 19, 5, 10}, {1, 32, 2, 8},  {4, 41, 6, 9},  {1, 512, 3, 2}, {1, 999, 2, 3},
};

const std::vector<bool> acrossChannels = {true, false};

const std::vector<bool> normalizeVariance = {true, false};

const std::vector<double> epsilon = {0.000000001};

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_3D, KmbMvnLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels3D),
                                            ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbMvnLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN, KmbMvnLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbMvnLayerTest::getTestCaseName);

// Test MVN MLIR

INSTANTIATE_TEST_CASE_P(smoke_TestsMVN_MLIR, KmbMvnLayerTestMLIR,
                        ::testing::Combine(::testing::ValuesIn(MLIRinputShapes),
                                           ::testing::Values(InferenceEngine::Precision::FP32),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMvnLayerTestMLIR::getTestCaseName);

// Test MVN MLIR VPU3720

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsMVN_MLIR_VPU3720, KmbMvnLayerTestMLIR_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(MLIRinputShapes),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(HWLayout::NCHW), ::testing::ValuesIn(emptyReductionAxes),
                                           ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                                           ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMvnLayerTestMLIR_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsMVN_MLIR_VPU3720_order, KmbMvnLayerTestMLIR_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(MLIRinputShapesForOrder),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW, HWLayout::NCWH, HWLayout::NWHC}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMvnLayerTestMLIR_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestsMVN_MLIR_VPU3720, KmbMvnLayerTestMLIR_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapes3720),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::Values(false),
                                           ::testing::Values(true), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMvnLayerTestMLIR_VPU3720::getTestCaseName);

// Test MVN-6

const std::vector<std::string> epsMode = {"inside_sqrt", "outside_sqrt"};

const std::vector<float> epsilonF = {0.0001};

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_4D, KmbMvn6LayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}}),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(std::vector<std::vector<int>>{
                                                    {1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                                            ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilonF),
                                            ::testing::Values("outside_sqrt"),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbMvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MVN6_4D_MLIR, KmbMvn6LayerTestMLIR,
                        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}}),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Precision::I32),
                                           ::testing::ValuesIn(std::vector<std::vector<int>>{
                                                   {1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilonF),
                                           ::testing::Values("outside_sqrt"),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMvn6LayerTestMLIR::getTestCaseName);
