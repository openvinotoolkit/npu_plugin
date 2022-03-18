// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution_backprop_data.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbConvolutionBackpropDataLayerTest: public ConvolutionBackpropDataLayerTest,
                                               virtual public LayerTestsUtils::KmbLayerTestsCommon {
    };

    TEST_P(KmbConvolutionBackpropDataLayerTest, CompareWithRefs) {
        Run();
    }

    class KmbConvolutionBackpropDataLayerTest_MLIR: public ConvolutionBackpropDataLayerTest,
                                               virtual public LayerTestsUtils::KmbLayerTestsCommon {
    };

    TEST_P(KmbConvolutionBackpropDataLayerTest_MLIR, CompareWithRefs) {
        useCompilerMLIR();
        Run();
    }

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    /// Current Deconv impelmentation Only support 16x channels (MCM)
    /// The other channel which needs alignment and crop will cause concat issue
    const std::vector<size_t> numOutChannels = {16};
    const std::vector<size_t> specificNumOutChannels = {128};
    const std::vector<std::vector<size_t >> emptyOutputShape = {{}};
    const std::vector<std::vector<size_t >> outputShape = {{32, 64}};
    const std::vector<std::vector<ptrdiff_t >> emptyOutputPadding = {{}};

/* ============= 2D ConvolutionBackpropData ============= */
    const std::vector<std::vector<size_t >> inputShapes2D = {{1, 3, 30, 30}};
    const std::vector<std::vector<size_t >> inputShapes2D_MLIR = {{1, 3, 30, 30}, {1, 32, 23, 30}, {1, 32, 46, 60}, {1, 32, 92, 120}, {1, 32, 184, 240}};
    const std::vector<std::vector<size_t >> specificInputShapes2D_MLIR = {{1, 256, 16, 32}};
    /// Need Kernel_size == Stride_size (MCM)
    /// Refer: src/mcmCompiler/src/pass/adaptation/conv_dilation_pass.cpp:366
    const std::vector<std::vector<size_t >> kernels2D = {{2, 2}};
    const std::vector<std::vector<size_t >> specificKernels2D = {{4, 4}};
    const std::vector<std::vector<size_t >> strides2D = {{2, 2}};
    const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
    const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
    const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};

    /// Not support SAME_UPPER padding mode (MCM)
    /// Refer: src/mcmCompiler/src/pass/adaptation/conv_dilation_pass.cpp:61
    const auto conv2DParams_ExplicitPadding = ::testing::Combine(
            ::testing::ValuesIn(kernels2D),
            ::testing::ValuesIn(strides2D),
            ::testing::ValuesIn(padBegins2D),
            ::testing::ValuesIn(padEnds2D),
            ::testing::ValuesIn(dilations2D),
            ::testing::ValuesIn(numOutChannels),
            ::testing::Values(ngraph::op::PadType::EXPLICIT),
            ::testing::ValuesIn(emptyOutputPadding)
    );
    const auto conv2DParams_AutoPadValid = ::testing::Combine(
            ::testing::ValuesIn(kernels2D),
            ::testing::ValuesIn(strides2D),
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
            ::testing::ValuesIn(dilations2D),
            ::testing::ValuesIn(numOutChannels),
            ::testing::Values(ngraph::op::PadType::VALID),
            ::testing::ValuesIn(emptyOutputPadding)
    );
    const auto conv2DParams_AutoPadSameLower = ::testing::Combine(
            ::testing::ValuesIn(specificKernels2D),
            ::testing::ValuesIn(strides2D),
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
            ::testing::ValuesIn(dilations2D),
            ::testing::ValuesIn(specificNumOutChannels),
            ::testing::Values(ngraph::op::PadType::SAME_LOWER),
            ::testing::ValuesIn(emptyOutputPadding)
    );

    // Test-case fails at stage "Run MCM Compiler" with error:
    // vpuxFuncTests: kmb-plugin/src/mcmCompiler/src/scheduler/feasible_scheduler.hpp:2198:
    // void mv::lp_scheduler::Feasible_Memory_Schedule_Generator<T, SchedulerTraits, Allocator>::
    // unschedule_op(const mv::lp_scheduler::Feasible_Memory_Schedule_Generator<T, SchedulerTraits, Allocator>::
    // heap_element_t&) [with T = mv::scheduler::Operation_Dag<>; SchedulerTraits =
    // mv::lp_scheduler::scheduler_traits<mv::scheduler::Operation_Dag<> >;
    // Allocator = std::allocator<mv::scheduler::Operation_Dag<> >]: Assertion `itr != op_output_table_.end()' failed.
    // Aborted (core dumped)
    // [Track number: S#44901]
    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_ExplicitPadding, KmbConvolutionBackpropDataLayerTest,
                            ::testing::Combine(
                                    conv2DParams_ExplicitPadding,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(inputShapes2D),
                                    ::testing::ValuesIn(emptyOutputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_ExplicitPadding, KmbConvolutionBackpropDataLayerTest_MLIR,
                            ::testing::Combine(
                                    conv2DParams_ExplicitPadding,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(inputShapes2D_MLIR),
                                    ::testing::ValuesIn(emptyOutputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest_MLIR::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_OutputShape, KmbConvolutionBackpropDataLayerTest_MLIR,
                            ::testing::Combine(
                                    conv2DParams_AutoPadSameLower,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(specificInputShapes2D_MLIR),
                                    ::testing::ValuesIn(outputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest_MLIR::getTestCaseName);

    // Test-case fails at stage "Run MCM Compiler" with error:
    // vpuxFuncTests: kmb-plugin/src/mcmCompiler/src/scheduler/feasible_scheduler.hpp:2198:
    // void mv::lp_scheduler::Feasible_Memory_Schedule_Generator<T, SchedulerTraits, Allocator>::
    // unschedule_op(const mv::lp_scheduler::Feasible_Memory_Schedule_Generator<T, SchedulerTraits, Allocator>::
    // heap_element_t&) [with T = mv::scheduler::Operation_Dag<>; SchedulerTraits =
    // mv::lp_scheduler::scheduler_traits<mv::scheduler::Operation_Dag<> >;
    // Allocator = std::allocator<mv::scheduler::Operation_Dag<> >]: Assertion `itr != op_output_table_.end()' failed.
    // Aborted (core dumped)
    // [Track number: S#44901]
    INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvolutionBackpropData2D_AutoPadValid, KmbConvolutionBackpropDataLayerTest,
                            ::testing::Combine(
                                    conv2DParams_AutoPadValid,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(inputShapes2D),
                                    ::testing::ValuesIn(emptyOutputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_AutoPadValid, KmbConvolutionBackpropDataLayerTest_MLIR,
                            ::testing::Combine(
                                    conv2DParams_AutoPadValid,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(inputShapes2D_MLIR),
                                    ::testing::ValuesIn(emptyOutputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest_MLIR::getTestCaseName);

/* ============= 3D ConvolutionBackpropData ============= */
    const std::vector<std::vector<size_t >> inputShapes3D = {{1, 3, 10, 10, 10},
                                                             {1, 16, 5, 5, 5},
                                                             {1, 32, 5, 5, 5}};
    const std::vector<std::vector<size_t >> kernels3D = {{1, 1, 1}, {3, 3, 3}};
    const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1}};
    const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
    const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}, {1, 1, 1}};
    const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}, {2, 2, 2}};

    const auto conv3DParams_ExplicitPadding = ::testing::Combine(
            ::testing::ValuesIn(kernels3D),
            ::testing::ValuesIn(strides3D),
            ::testing::ValuesIn(padBegins3D),
            ::testing::ValuesIn(padEnds3D),
            ::testing::ValuesIn(dilations3D),
            ::testing::ValuesIn(numOutChannels),
            ::testing::Values(ngraph::op::PadType::EXPLICIT),
            ::testing::ValuesIn(emptyOutputPadding)
    );
    const auto conv3DParams_AutoPadValid = ::testing::Combine(
            ::testing::ValuesIn(kernels3D),
            ::testing::ValuesIn(strides3D),
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
            ::testing::ValuesIn(dilations3D),
            ::testing::ValuesIn(numOutChannels),
            ::testing::Values(ngraph::op::PadType::VALID),
            ::testing::ValuesIn(emptyOutputPadding)
    );

    // All test instances fail at stage "Convert nGraph to MCM Model" with error:
    // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
    // doesn't throw an exception.
    // Actual: it throws:Unsupported dimensions layout
    // kmb-plugin/src/utils/dims_parser.cpp:45
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64
    // [Track number: S#44901]
    INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvolutionBackpropData3D_ExplicitPadding, KmbConvolutionBackpropDataLayerTest,
                            ::testing::Combine(
                                    conv3DParams_ExplicitPadding,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(inputShapes3D),
                                    ::testing::ValuesIn(emptyOutputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest::getTestCaseName);

    // All test instances fail at stage "Convert nGraph to MCM Model" with error:
    // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
    // doesn't throw an exception.
    // Actual: it throws:Unsupported dimensions layout
    // kmb-plugin/src/utils/dims_parser.cpp:45
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64
    // [Track number: S#44901]
    INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvolutionBackpropData3D_AutoPadValid, KmbConvolutionBackpropDataLayerTest,
                            ::testing::Combine(
                                    conv3DParams_AutoPadValid,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::ValuesIn(inputShapes3D),
                                    ::testing::ValuesIn(emptyOutputShape),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConvolutionBackpropDataLayerTest::getTestCaseName);

}  // namespace
