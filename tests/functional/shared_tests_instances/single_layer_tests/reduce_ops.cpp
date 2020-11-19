// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reduce_ops.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbReduceOpsLayerTest : public ReduceOpsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeImport() override {
            throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
        }
        void SkipBeforeValidate() override {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    };

    TEST_P(KmbReduceOpsLayerTest, CompareWithRefs) {
        Run();
    }

    class KmbReduceOpsLayerWithSpecificInputTest : public ReduceOpsLayerWithSpecificInputTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeImport() override {
            throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
        }
        void SkipBeforeValidate() override {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    };

    TEST_P(KmbReduceOpsLayerWithSpecificInputTest, CompareWithRefs) {
        Run();
    }

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16, // CPU-plugin has parameter I32, but KMB does not
            InferenceEngine::Precision::U8    // support it. So I32 is changed to FP16 and U8.
    };

    const std::vector<bool> keepDims = {
            true,
            false,
    };

    const std::vector<std::vector<size_t>> inputShapes = {
            std::vector<size_t>{10, 20, 30, 40},
            std::vector<size_t>{3, 5, 7, 9},
    };

    const std::vector<std::vector<size_t>> inputShapesOneAxis = {
            std::vector<size_t>{10, 20, 30, 40},
            std::vector<size_t>{3, 5, 7, 9},
            std::vector<size_t>{10},
    };

    const std::vector<std::vector<int>> axes = {
            {0},
            {1},
            {2},
            {3},
            {0, 1},
            {0, 2},
            {0, 3},
            {1, 2},
            {1, 3},
            {2, 3},
            {0, 1, 2},
            {0, 1, 3},
            {0, 2, 3},
            {1, 2, 3},
            {0, 1, 2, 3},
            {1, -1}
    };

    std::vector<CommonTestUtils::OpType> opTypes = {
            CommonTestUtils::OpType::SCALAR,
            CommonTestUtils::OpType::VECTOR,
    };

    const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
            ngraph::helpers::ReductionType::Mean,
            ngraph::helpers::ReductionType::Min,
            ngraph::helpers::ReductionType::Max,
            ngraph::helpers::ReductionType::Sum,
            ngraph::helpers::ReductionType::Prod,
            ngraph::helpers::ReductionType::LogicalOr,
            ngraph::helpers::ReductionType::LogicalXor,
            ngraph::helpers::ReductionType::LogicalAnd,
    };

    const auto paramsOneAxis = testing::Combine(
            testing::Values(std::vector<int>{0}),
            testing::ValuesIn(opTypes),
            testing::Values(true, false),
            testing::ValuesIn(reductionTypes),
            testing::Values(InferenceEngine::Precision::FP32),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapesOneAxis),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    const auto params_Precisions = testing::Combine(
            testing::Values(std::vector<int>{1, 3}),
            testing::Values(opTypes[1]),
            testing::ValuesIn(keepDims),
            testing::Values(ngraph::helpers::ReductionType::Sum),
            testing::Values(InferenceEngine::Precision::FP32,
                            InferenceEngine::Precision::FP16, // CPU-plugin has parameter I32, but KMB does not
                            InferenceEngine::Precision::U8), // support it. So I32 is changed to FP16 and U8.
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 2, 2, 2}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    const auto params_InputShapes = testing::Combine(
            testing::Values(std::vector<int>{0}),
            testing::Values(opTypes[1]),
            testing::ValuesIn(keepDims),
            testing::Values(ngraph::helpers::ReductionType::Mean),
            testing::Values(InferenceEngine::Precision::FP32),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{3},
                            std::vector<size_t>{3, 5},
                            std::vector<size_t>{2, 4, 6},
                            std::vector<size_t>{2, 4, 6, 8},
                            std::vector<size_t>{2, 2, 2, 2, 2},
                            std::vector<size_t>{2, 2, 2, 2, 2, 2}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    const auto params_Axes = testing::Combine(
            testing::ValuesIn(axes),
            testing::Values(opTypes[1]),
            testing::ValuesIn(keepDims),
            testing::Values(ngraph::helpers::ReductionType::Mean),
            testing::Values(InferenceEngine::Precision::FP32),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapes),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    const auto params_ReductionTypes = testing::Combine(
            testing::Values(std::vector<int>{0, 1, 3}),
            testing::Values(opTypes[1]),
            testing::ValuesIn(keepDims),
            testing::ValuesIn(reductionTypes),
            testing::Values(InferenceEngine::Precision::FP32),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 9, 2, 9}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    // Tests fail with one of typical error:
    // 1. C++ exception with description "GraphOptimizer-StrategyManager - LogicError: No strategies created for
    // layer ReduceMean_16565/pool_DepthwiseConv. Layer possibly unsupported.
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 2. C++ exception with description "GraphOptimizer-StrategyManager - LogicError: No strategies created for
    // layer ReduceMax_2195/pool. Layer possibly unsupported.
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 3. C++ exception with description "GraphOptimizer-StrategyManager - LogicError: No strategies created for
    // layer ReduceSum_3299/pool_DepthwiseConv. Layer possibly unsupported.
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 4. C++ exception with description "Unsupported operation: ReduceMin_1106 with name ReduceMin_1123 with
    // type ReduceMin with C++ type N6ngraph2op2v19ReduceMinE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 5. C++ exception with description "Unsupported operation: ReduceProd_4409 with name ReduceProd_4426 with
    // type ReduceProd with C++ type N6ngraph2op2v110ReduceProdE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 6. C++ exception with description "Check 'element::Type::merge(element_type, element_type,
    // node->get_input_element_type(i))' failed at ngraph/core/src/op/util/elementwise_args.cpp:37:
    // While validating node 'v1::LogicalOr LogicalOr_5498 (Parameter_5496[0]:f32{10,20,30,40},
    // Constant_5497[0]:i64{}) -> (dynamic?)' with friendly_name 'LogicalOr_5498':
    // Argument element types are inconsistent." thrown in SetUp().
    //
    // 7. C++ exception with description "Output layout is not supported: 95
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:239
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // [Track number: S#43428]
    INSTANTIATE_TEST_CASE_P(
            DISABLED_smoke_ReduceOneAxis,
            KmbReduceOpsLayerTest,
            paramsOneAxis,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // All tests in this test case fail with the same error:
    // C++ exception with description "Unsupported operation: ReduceSum_365 with name ReduceSum_382 with type ReduceSum with C++ type N6ngraph2op2v19ReduceSumE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    // [Track number: S#43428]
    INSTANTIATE_TEST_CASE_P(
            DISABLED_smoke_Reduce_Precisions,
            KmbReduceOpsLayerTest,
            params_Precisions,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // Tests in this test case fail with one of typical error:
    // 1. C++ exception with description "Unsupported dimensions layout
    // kmb-plugin/src/utils/dims_parser.cpp:45
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 2. C++ exception with description "Output layout is not supported: 95
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:239
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    // [Track number: S#43428]
    INSTANTIATE_TEST_CASE_P(
            DISABLED_smoke_Reduce_InputShapes,
            KmbReduceOpsLayerTest,
            params_InputShapes,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // Test hangs on x86 when executes test-case
    // smoke_Reduce_Axes/KmbReduceOpsLayerTest.CompareWithRefs/IS=(10.20.30.40)_axes=(1)_opType=VECTOR_
    // type=Mean_KeepDims_netPRC=FP32_inPRC=UNSPECIFIED_outPRC=UNSPECIFIED_inL=ANY_trgDev=KMB
    // [Track number: S#43428]
    INSTANTIATE_TEST_CASE_P(
            DISABLED_smoke_Reduce_Axes,
            KmbReduceOpsLayerTest,
            params_Axes,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // Tests fails with one of typical error:
    // 1. C++ exception with description "Unsupported operation: ReduceMean_2 with name ReduceMean_19 with
    // type ReduceMean with C++ type N6ngraph2op2v110ReduceMeanE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 2. C++ exception with description "Unsupported operation: ReduceMin_365 with name ReduceMin_382 with
    // type ReduceMin with C++ type N6ngraph2op2v19ReduceMinE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 3. C++ exception with description "Unsupported operation: ReduceMax_728 with name ReduceMax_745 with
    // type ReduceMax with C++ type N6ngraph2op2v19ReduceMaxE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 4. C++ exception with description "Unsupported operation: ReduceSum_1091 with name ReduceSum_1108 with
    // type ReduceSum with C++ type N6ngraph2op2v19ReduceSumE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 5. C++ exception with description "Unsupported operation: ReduceProd_1454 with name ReduceProd_1471 with
    // type ReduceProd with C++ type N6ngraph2op2v110ReduceProdE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 6. C++ exception with description "Check 'element::Type::merge(element_type, element_type,
    // node->get_input_element_type(i))' failed at ngraph/core/src/op/util/elementwise_args.cpp:37:
    // While validating node 'v1::LogicalOr LogicalOr_1817 (Parameter_1815[0]:f32{2,9,2,9},
    // Constant_1816[0]:i64{3}) -> (dynamic?)' with friendly_name 'LogicalOr_1817':
    // Argument element types are inconsistent." thrown in SetUp().
    //
    // [Track number: S#43428]
    INSTANTIATE_TEST_CASE_P(
            DISABLED_smoke_Reduce_ReductionTypes,
            KmbReduceOpsLayerTest,
            params_ReductionTypes,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // Tests fails with one of typical error:
    // 1. C++ exception with description "MemoryAllocator:VPU_DDR_Heap - ArgumentError:
    // ReduceSum_372/mul_DepthwiseConv_copyOutReduceSum_372/mul_DepthwiseConv:0:0::paddedShape[0] 10 -
    // Does not match the dimension 1 of the tensor ImplicitReshape_0:0 already allocated in the given buffer
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 2. C++ exception with description "Unsupported operation: Convert_1105 with name Convert_1105 with
    // type Convert with C++ type N6ngraph2op2v07ConvertE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // [Track number: S#43428]
    INSTANTIATE_TEST_CASE_P(
            DISABLED_smoke_Reduce,
            KmbReduceOpsLayerWithSpecificInputTest,
            testing::Combine(
                    testing::ValuesIn(decltype(axes) {{0}, {1}}),
                    testing::Values(opTypes[1]),
                    testing::Values(true),
                    testing::Values(ngraph::helpers::ReductionType::Sum),
                    testing::Values(InferenceEngine::Precision::FP32,
                                    InferenceEngine::Precision::FP16, // I32 is not supported by KMB.
                                    InferenceEngine::Precision::U8),  // It is changed to FP16 & U8.
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Layout::ANY),
                    testing::Values(std::vector<size_t> {2, 10}),
                    testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerWithSpecificInputTest::getTestCaseName
    );

    // Subset of parameters and test-cases below are just temporal variants to run in time of
    // debugging initial tests. Do not forget to remove them when initial tests will be enabled.
    const std::vector<std::vector<size_t>> inputShapesOneAxis_pass_mcm = {
            std::vector<size_t>{3, 5, 7, 9},
    };

    const std::vector<ngraph::helpers::ReductionType> reductionTypes_pass_mcm = {
            ngraph::helpers::ReductionType::Mean,
            ngraph::helpers::ReductionType::Max,
            ngraph::helpers::ReductionType::Sum,
    };

    const auto paramsOneAxis_pass_mcm = testing::Combine(
            testing::Values(std::vector<int>{0}),
            testing::ValuesIn(opTypes),
            testing::Values(true, false),
            testing::ValuesIn(reductionTypes_pass_mcm),
            testing::Values(InferenceEngine::Precision::FP32),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapesOneAxis_pass_mcm),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_CASE_P(
            smoke_ReduceOneAxis_pass_mcm,
            KmbReduceOpsLayerTest,
            paramsOneAxis_pass_mcm,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    const auto params_InputShapes_pass_mcm = testing::Combine(
            testing::Values(std::vector<int>{0}),
            testing::Values(opTypes[1]),
            testing::ValuesIn(keepDims),
            testing::Values(ngraph::helpers::ReductionType::Mean),
            testing::Values(InferenceEngine::Precision::FP32),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{3, 5},
                            std::vector<size_t>{2, 4, 6},
                            std::vector<size_t>{2, 4, 6, 8}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_CASE_P(
            smoke_Reduce_InputShapes_pass_mcm,
            KmbReduceOpsLayerTest,
            params_InputShapes_pass_mcm,
            KmbReduceOpsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_CASE_P(
            smoke_Reduce_pass_mcm,
            KmbReduceOpsLayerWithSpecificInputTest,
            testing::Combine(
                    testing::ValuesIn(decltype(axes) { {1} }),
                    testing::Values(opTypes[1]),
                    testing::Values(true),
                    testing::Values(ngraph::helpers::ReductionType::Sum),
                    testing::Values(InferenceEngine::Precision::FP32,
                                    InferenceEngine::Precision::FP16), 
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Layout::ANY),
                    testing::Values(std::vector<size_t> {2, 10}),
                    testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerWithSpecificInputTest::getTestCaseName
    );

    // end of temporal tests

}  // namespace
