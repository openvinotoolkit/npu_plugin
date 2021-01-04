//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <memory>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

typedef std::tuple<std::vector<size_t>,  // Input shape 1
                   std::vector<size_t>,  // Input shape 2
                   bool,                 // transpose 1
                   bool                  // transpose 2
                   >
        matMulSpecificParams;

class IE_FrontEndTest_MatMulLayer : public testing::TestWithParam<matMulSpecificParams> {};

TEST_P(IE_FrontEndTest_MatMulLayer, MatMulLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        ngraph::Shape input1, input2;
        bool transposeA, transposeB;

        std::tie(input1, input2, transposeA, transposeB) = this->GetParam();

        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input1);
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input2);

        auto matmul = std::make_shared<ngraph::opset1::MatMul>(param1, param2, transposeA, transposeB);
        matmul->set_friendly_name("MatMul");
        auto result = std::make_shared<ngraph::op::Result>(matmul);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    auto module = vpux::IE::importNetwork(&ctx, nGraphImpl);
    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl));
}

/* +========== Test for Matrix x Matrix ========== */
const auto matMul_MatMat_Param = ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{2, 3}),  // input1
                                                    ::testing::Values(InferenceEngine::SizeVector{3, 4}),  // input2
                                                    ::testing::Values(false),   // transpose_a
                                                    ::testing::Values(false));  // transpose_b

INSTANTIATE_TEST_CASE_P(MatMat, IE_FrontEndTest_MatMulLayer, matMul_MatMat_Param);

/* +========== Test for Matrix x Matrix with Batch ========== */
const auto matMul_MatMatBatch_Param =
        ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{4, 8, 2, 3}),  // input1
                           ::testing::Values(InferenceEngine::SizeVector{8, 3, 4}),     // input2
                           ::testing::Values(false),                                    // transpose_a
                           ::testing::Values(false));                                   // transpose_b

INSTANTIATE_TEST_CASE_P(MatMatBatch, IE_FrontEndTest_MatMulLayer, matMul_MatMatBatch_Param);

/* +========== Test for Matrix x Vector ========== */
const auto matMul_MatVec_Param = ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{2, 3}),  // input1
                                                    ::testing::Values(InferenceEngine::SizeVector{3}),     // input2
                                                    ::testing::Values(false),   // transpose_a
                                                    ::testing::Values(false));  // transpose_b

INSTANTIATE_TEST_CASE_P(MatVec, IE_FrontEndTest_MatMulLayer, matMul_MatVec_Param);

/* +========== Test for Vector x Matrix  ========== */
const auto matMul_VecMat_Param = ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{5}),      // input1
                                                    ::testing::Values(InferenceEngine::SizeVector{5, 11}),  // input2
                                                    ::testing::Values(false),   // transpose_a
                                                    ::testing::Values(false));  // transpose_b

INSTANTIATE_TEST_CASE_P(VecMat, IE_FrontEndTest_MatMulLayer, matMul_VecMat_Param);

/* +========== Test for Vector x Vector ========== */
const auto matMul_Scalar_Param = ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{5}),  // input1
                                                    ::testing::Values(InferenceEngine::SizeVector{5}),  // input2
                                                    ::testing::Values(false),                           // transpose_a
                                                    ::testing::Values(false));                          // transpose_b

INSTANTIATE_TEST_CASE_P(Scalar, IE_FrontEndTest_MatMulLayer, matMul_Scalar_Param);

/* +========== Test for Matrix x Matrix with tranpose A and batch ========== */
const auto matMul_TransposeA_Param =
        ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{2, 11, 3}),  // input1
                           ::testing::Values(InferenceEngine::SizeVector{11, 5}),     // input2
                           ::testing::Values(true),                                   // transpose_a
                           ::testing::Values(false));                                 // transpose_b

INSTANTIATE_TEST_CASE_P(TransposeA, IE_FrontEndTest_MatMulLayer, matMul_TransposeA_Param);

/* +========== Test for Matrix x Matrix with tranpose both A and B and batch ========== */
const auto matMul_TransposeAB_Param =
        ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{2, 11, 3}),  // input1
                           ::testing::Values(InferenceEngine::SizeVector{2, 5, 11}),  // input2
                           ::testing::Values(true),                                   // transpose_a
                           ::testing::Values(true));                                  // transpose_b

INSTANTIATE_TEST_CASE_P(TransposeAB, IE_FrontEndTest_MatMulLayer, matMul_TransposeAB_Param);

/* +========== Test for Matrix x Matrix with tranpose both A and B which is vector ========== */
const auto matMul_TransposeABVec_Param =
        ::testing::Combine(::testing::Values(InferenceEngine::SizeVector{2, 11, 3}),  // input1
                           ::testing::Values(InferenceEngine::SizeVector{11}),        // input2
                           ::testing::Values(true),                                   // transpose_a
                           ::testing::Values(true));                                  // transpose_b

INSTANTIATE_TEST_CASE_P(TransposeABVec, IE_FrontEndTest_MatMulLayer, matMul_TransposeABVec_Param);
