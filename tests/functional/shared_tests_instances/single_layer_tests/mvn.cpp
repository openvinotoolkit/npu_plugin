//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <ngraph/op/parameter.hpp>
#include <openvino/op/mvn.hpp>
#include <ov_models/builders.hpp>
#include <shared_test_classes/single_layer/mvn.hpp>
#include <single_layer_tests/mvn.hpp>
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_properties.hpp"

namespace LayerTestsDefinitions {

// -------------- MVN1 test classes

class Mvn1LayerTest_NPU3700 : public Mvn1LayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

TEST_P(Mvn1LayerTest_NPU3700, HW) {
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
class Mvn1LayerTest_NPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
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

        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ngraph::element::f16, ov::Shape(inputShapes))};
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        // order handling
        auto input_transpose_order = ov::op::v0::Constant::create(ngraph::element::i64, {4}, make_transpose_order());
        auto input_transpose = std::make_shared<ov::op::v1::Transpose>(paramOuts[0], input_transpose_order);

        auto make_mvn_kernel = [&]() {
            if (!axes.empty()) {
                return std::dynamic_pointer_cast<ngraph::op::MVN>(
                        ngraph::builder::makeMVN(input_transpose, axes, normalizeVariance, eps));
            }
            return std::dynamic_pointer_cast<ngraph::op::MVN>(
                    ngraph::builder::makeMVN(input_transpose, acrossChanels, normalizeVariance, eps));
        };
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(make_mvn_kernel())};
        function = std::make_shared<ngraph::Function>(results, params, "MVN1");
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

/**
 * MVN1 Input=0 special case E#96869
 */
class Mvn1ZeroInputLayerTestCommon : public Mvn1LayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& inputInfo) const override {
        return FuncTestUtils::createAndFillBlob(inputInfo.getTensorDesc(), 0, 0, 1, 0);
    }
};

class Mvn1ZeroInputLayerTest_NPU3720 : public Mvn1ZeroInputLayerTestCommon {};

TEST_P(Mvn1LayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(Mvn1ZeroInputLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

// -------------- MVN6 test classes

class Mvn6LayerTestCommon : public Mvn6LayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    // Avoid f32->f16 converts
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }
};

class Mvn6LayerTest_NPU3700 : public Mvn6LayerTestCommon {};
class Mvn6LayerTest_NPU3720 : public Mvn6LayerTestCommon {};

class Mvn6LayerTest_FP32 : public Mvn6LayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void ConfigureNetwork() override {
        configuration[ov::intel_vpux::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(Mvn6LayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(Mvn6LayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

// -------------- MVN6 F32 tests

TEST_P(Mvn6LayerTest_FP32, NPU3720) {
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

/* ================================= MVN1 NPU3700 ================================= */

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
// Currently input dim > 4 is not supported by NPU-plugin
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
#endif
};

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_3D, Mvn1LayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels3D),
                                            ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         Mvn1LayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN, Mvn1LayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         Mvn1LayerTest_NPU3700::getTestCaseName);

// MVN6 pseudo test (actually testing MVN1 op, as innermost-consecutive norm axes config trigger 'ConvertMVN6toMVN1')
// MVN6 kernel not available on NPU3700
INSTANTIATE_TEST_CASE_P(pseudo_MVN6_4D_MLIR, Mvn6LayerTest_NPU3700,
                        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}}),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Precision::I32),
                                           ::testing::ValuesIn(std::vector<std::vector<int>>{
                                                   {1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilonF),
                                           ::testing::Values("outside_sqrt"),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Mvn6LayerTest_NPU3700::getTestCaseName);

/* ================================= Param builder utils ================================= */

const auto genMvn6Params = [](auto shapes, auto axes, auto eps) {
    return ::testing::Combine(::testing::ValuesIn(shapes), ::testing::Values(InferenceEngine::Precision::FP16),
                              ::testing::Values(InferenceEngine::Precision::I32), ::testing::ValuesIn(axes),
                              ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(eps),
                              ::testing::ValuesIn(epsMode),
                              ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
};

// less test combinations
const auto genMvn6LessParams = [](auto shape, auto axes, auto eps) {
    bool normVariance = true;  // typical configs
    const std::string epsMode = "inside_sqrt";
    return ::testing::Combine(::testing::Values(shape), ::testing::Values(InferenceEngine::Precision::FP16),
                              ::testing::Values(InferenceEngine::Precision::I32), ::testing::ValuesIn(axes),
                              ::testing::Values(normVariance), ::testing::ValuesIn(eps), ::testing::Values(epsMode),
                              ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
};

/* ============================ MVN1 tests (NPU3720) ============================= */

const std::vector<std::vector<size_t>> inputShapesForOrder = {{1, 4, 2, 1024}};

const std::vector<std::vector<size_t>> inputShapes4D = {{1, 4, 512, 1}, {1, 999, 2, 3}, {1, 16, 5, 8}, {2, 19, 5, 10},
                                                        {7, 32, 2, 8},  {5, 8, 3, 5},   {4, 41, 6, 9}
#if 0  // extra shapes
        {5, 2, 7, 3},   {1, 3, 17, 21}, {2, 5, 13, 27}, {1, 7, 55, 33}, {4, 9, 7, 2},  {3, 13, 9, 9}, {1, 16, 12, 11},
        {1, 512, 3, 2},
#endif
};
const std::vector<std::vector<size_t>> inputShapesForDecomposition = {{1, 1, 1, 515971}, {2, 3, 20, 35971}};
const std::vector<std::vector<size_t>> inputShapesForNHWCOpt = {{1, 16, 4, 32}, {1, 32, 4, 16}};
const std::vector<std::vector<size_t>> inputShapesForBigSize = {{1, 1, 8, 48}, {1, 2, 8, 128}};

// -------------- MVN1 - NPU3270

INSTANTIATE_TEST_CASE_P(precommit_MVN1_order, Mvn1LayerTest_NPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapesForOrder),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW, HWLayout::NCWH, HWLayout::NWHC}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Mvn1LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MVN1, Mvn1LayerTest_NPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapes4D),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Mvn1LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(precommit_MVN1_opt, Mvn1LayerTest_NPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapesForNHWCOpt),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NHWC}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Mvn1LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(precommit_MVN1_bigsize, Mvn1LayerTest_NPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapesForBigSize),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Mvn1LayerTest_NPU3720::getTestCaseName);

// -------------- MVN1 Decomposition

INSTANTIATE_TEST_CASE_P(smoke_MVN1_Decomposition, Mvn1LayerTest_NPU3720,
                        ::testing::Combine(::testing::ValuesIn(inputShapesForDecomposition),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::ValuesIn({HWLayout::NCHW}),
                                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Mvn1LayerTest_NPU3720::getTestCaseName);

// -------------- MVN6 'pseudo' tests : actually testing MVN1 op,
// as innermost-consecutive norm axes config trigger 'ConvertMVN6toMVN1' to pass

const auto pse2D = genMvn6LessParams(std::vector<size_t>{5, 17}, AxesVec{{1}}, epsilonF);
const auto pse3D = genMvn6LessParams(std::vector<size_t>{10, 5, 17}, AxesVec{{2}}, epsilonF);

INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_2D, Mvn6LayerTest_NPU3720, pse2D, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_3D, Mvn6LayerTest_NPU3720, pse3D, Mvn6LayerTest_NPU3720::getTestCaseName);

// -------------- MVN1 Zero-Input test

const auto zeroTestCfg = ::testing::Combine(
        ::testing::Values(inputShapes4D[0]), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels), ::testing::Values(true),
        ::testing::ValuesIn(epsilon), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(zero_input, Mvn1ZeroInputLayerTest_NPU3720, zeroTestCfg,
                        Mvn1ZeroInputLayerTest_NPU3720::getTestCaseName);

/* ============================= MVN6 tests (NPU3720) ============================ */

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

// -------------- MVN6 - NPU3720

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_1D, Mvn6LayerTest_NPU3720, cfg1D, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_2D, Mvn6LayerTest_NPU3720, cfg2D, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_3D, Mvn6LayerTest_NPU3720, cfg3D, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_4D, Mvn6LayerTest_NPU3720, cfg4D, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_5D, Mvn6LayerTest_NPU3720, cfg5D, Mvn6LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(tiling_MVN6_a, Mvn6LayerTest_NPU3720, cfgT0, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(tiling_MVN6_b, Mvn6LayerTest_NPU3720, cfgT1, Mvn6LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(tiling_MVN6_c, Mvn6LayerTest_NPU3720, cfgT2, Mvn6LayerTest_NPU3720::getTestCaseName);

// -------------- MVN6 - f32 tests
const auto cfgF32 =
        ::testing::Combine(::testing::Values(shapes4D[0]), ::testing::Values(InferenceEngine::Precision::FP32),
                           ::testing::Values(InferenceEngine::Precision::I32), ::testing::Values(axes4D[2]),
                           ::testing::Values(true), ::testing::ValuesIn(bigEps), ::testing::ValuesIn(epsMode),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_fp32, Mvn6LayerTest_FP32, cfgF32, Mvn6LayerTest_FP32::getTestCaseName);

}  // namespace
