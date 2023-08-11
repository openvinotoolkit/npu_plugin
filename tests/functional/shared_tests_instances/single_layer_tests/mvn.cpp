//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/parameter.hpp>
#include <ngraph_functions/builders.hpp>
#include <openvino/op/mvn.hpp>
#include <shared_test_classes/single_layer/mvn.hpp>
#include <single_layer_tests/mvn.hpp>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

// -------------- MVN1 test classes

class VPUXMvn1LayerTest_VPU3700 : public Mvn1LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(VPUXMvn1LayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
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
        mvn2Params;  // MVN1 with layout params

/**
 * For testing particular kernel with variable order currently there is no way to generalize that in KMBLayerTestCommon
 */
class VPUXMvn1LayerTest_VPU3720 :
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

TEST_P(VPUXMvn1LayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

// -------------- MVN6 test classes

class VPUXMvn6LayerTest : public Mvn6LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    // Avoid f32->f16 converts
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }
};
class VPUXMvn6LayerTest_VPU3700 : public VPUXMvn6LayerTest {};
class VPUXMvn6LayerTest_VPU3720 : public VPUXMvn6LayerTest {};

TEST_P(VPUXMvn6LayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXMvn6LayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

/* ================================= Common params ================================ */

const std::vector<bool> acrossChannels = {true, false};
const std::vector<bool> normalizeVariance = {true, false};
const std::vector<double> epsilon = {0.000000001};
const std::vector<float> epsilonF = {0.0001};
const std::vector<std::string> epsMode = {"inside_sqrt", "outside_sqrt"};
const std::vector<ngraph::AxisSet> emptyReductionAxes = {{}};
using AxesVec = std::vector<std::vector<int>>;

/* ================================= MVN1 VPU3700 ================================= */

// Accuracy problem for 3-dim input tensor when acrossChannels = false
// Traced in Jira S#48579
const std::vector<std::vector<size_t>> inputShapes3D = {{1, 32, 17}, {1, 37, 9}};

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
// Currently input dim > 4 is not supported by VPUX-plugin
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
#endif
};

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_3D, VPUXMvn1LayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels3D),
                                            ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXMvn1LayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN, VPUXMvn1LayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXMvn1LayerTest_VPU3700::getTestCaseName);

// MVN6 pseudo test (actually testing MVN1 op, as innermost-consecutive norm axes config trigger 'ConvertMVN6toMVN1')
// MVN6 kernel not available on VPU3700
INSTANTIATE_TEST_CASE_P(pseudo_MVN6_4D_MLIR, VPUXMvn6LayerTest_VPU3700,
                        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}}),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Precision::I32),
                                           ::testing::ValuesIn(std::vector<std::vector<int>>{
                                                   {1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilonF),
                                           ::testing::Values("outside_sqrt"),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXMvn6LayerTest_VPU3700::getTestCaseName);

/* ================================= Param builder utils ================================= */

const auto genMvn6Params = [](auto shapes, auto axes, auto eps) {
    return ::testing::Combine(::testing::ValuesIn(shapes), ::testing::Values(InferenceEngine::Precision::FP16),
                              ::testing::Values(InferenceEngine::Precision::I32), ::testing::ValuesIn(axes),
                              ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(eps),
                              ::testing::ValuesIn(epsMode),
                              ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));
};

// less test combinations
const auto genMvn6LessParams = [](auto shape, auto axes, auto eps) {
    bool normVariance = true;  // typical configs
    const std::string epsMode = "inside_sqrt";
    return ::testing::Combine(::testing::Values(shape), ::testing::Values(InferenceEngine::Precision::FP16),
                              ::testing::Values(InferenceEngine::Precision::I32), ::testing::ValuesIn(axes),
                              ::testing::Values(normVariance), ::testing::ValuesIn(eps), ::testing::Values(epsMode),
                              ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));
};

/* ============================ MVN1 tests (VPU3720) ============================= */

const std::vector<std::vector<size_t>> inputShapesForOrder = {{1, 4, 2, 1024}};

const std::vector<std::vector<size_t>> inputShapes4D = {
        {1, 4, 512, 1}, {1, 999, 2, 3}, {1, 16, 5, 8}, {2, 19, 5, 10}, {7, 32, 2, 8}, {5, 8, 3, 5}, {4, 41, 6, 9},
};

// -------------- MVN1 - VPU3270

INSTANTIATE_TEST_CASE_P(smoke_MVN1_order, VPUXMvn1LayerTest_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapesForOrder),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW, HWLayout::NCWH, HWLayout::NWHC}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXMvn1LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MVN1, VPUXMvn1LayerTest_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapes4D),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXMvn1LayerTest_VPU3720::getTestCaseName);

// -------------- MVN6 'pseudo' tests : actually testing MVN1 op,
// as innermost-consecutive norm axes config trigger 'ConvertMVN6toMVN1' to pass

const auto pse2D = genMvn6LessParams(std::vector<size_t>{5, 17}, AxesVec{{1}}, epsilonF);
const auto pse3D = genMvn6LessParams(std::vector<size_t>{10, 5, 17}, AxesVec{{2}}, epsilonF);

INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_2D, VPUXMvn6LayerTest_VPU3720, pse2D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_3D, VPUXMvn6LayerTest_VPU3720, pse3D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);

/* ============================= MVN6 tests (VPU3720) ============================ */

const std::vector<float> bigEps = {0.5};
std::vector<InferenceEngine::SizeVector> shapes1D = {{17}};
std::vector<InferenceEngine::SizeVector> shapes2D = {{5, 17}};
std::vector<InferenceEngine::SizeVector> shapes3D = {{10, 5, 17}};
std::vector<InferenceEngine::SizeVector> shapes4D = {{4, 10, 5, 17}};
std::vector<InferenceEngine::SizeVector> shapes5D = {{10, 16, 5, 10, 6}};

// axes values for corresponding ND shapes
std::vector<std::vector<int>> axes1D = {{0}};
std::vector<std::vector<int>> axes2D = {{1}, {0, 1}};
std::vector<std::vector<int>> axes3D = {{0}, {1}, {0, 1}};
std::vector<std::vector<int>> axes4D = {{0},    {1},    {2},       {3},       {0, 1},      {0, 2},
                                        {1, 2}, {1, 3}, {0, 1, 2}, {0, 1, 3}, {0, 1, 2, 3}};
std::vector<std::vector<int>> axes5D = {{1}, {1, 2}, {1, 3, 4}, {0, 2, 3}, {0, 1, 3, 4}};

// ND-shape configs
const auto cfg1D = genMvn6Params(shapes1D, axes1D, bigEps);
const auto cfg2D = genMvn6Params(shapes2D, axes2D, bigEps);
const auto cfg3D = genMvn6Params(shapes3D, axes3D, bigEps);
const auto cfg4D = genMvn6LessParams(shapes4D[0], axes4D, bigEps);
const auto cfg5D = genMvn6LessParams(shapes5D[0], axes5D, bigEps);

// tiling configs
const auto cfgT0 = genMvn6LessParams(InferenceEngine::SizeVector{1, 512, 1219}, AxesVec{{1}}, bigEps);
const auto cfgT1 = genMvn6LessParams(InferenceEngine::SizeVector{1, 64, 104, 104}, AxesVec{{2}}, bigEps);
const auto cfgT2 = genMvn6LessParams(InferenceEngine::SizeVector{8, 16, 16, 16, 16}, AxesVec{{4}}, bigEps);

// -------------- MVN6 - VPU3720

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_1D, VPUXMvn6LayerTest_VPU3720, cfg1D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_2D, VPUXMvn6LayerTest_VPU3720, cfg2D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_3D, VPUXMvn6LayerTest_VPU3720, cfg3D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_4D, VPUXMvn6LayerTest_VPU3720, cfg4D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_5D, VPUXMvn6LayerTest_VPU3720, cfg5D, VPUXMvn6LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(tiling_MVN6_a, VPUXMvn6LayerTest_VPU3720, cfgT0, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(tiling_MVN6_b, VPUXMvn6LayerTest_VPU3720, cfgT1, VPUXMvn6LayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(tiling_MVN6_c, VPUXMvn6LayerTest_VPU3720, cfgT2, VPUXMvn6LayerTest_VPU3720::getTestCaseName);

}  // namespace
