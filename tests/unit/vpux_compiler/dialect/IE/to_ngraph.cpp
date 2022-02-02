//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/backend/IE.hpp"
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include <vector>
#include <gtest/gtest.h>
#include <functional>
#include <utility>
#include <tuple>
#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/validation_util.hpp>
#include <legacy/ngraph_ops/lrn_ie.hpp>
#include <cpp/ie_cnn_network.h>
#include "legacy/ngraph_ops/normalize_ie.hpp"
#include <ngraph/opsets/opset4.hpp>

using namespace vpux;

class ToNgraphBackendTests : public testing::Test
{
public:
    mlir::MLIRContext ctx;
    template <typename T>
    std::shared_ptr<T> getNgraphNode(const llvm::StringLiteral inputIR);
    std::shared_ptr<ngraph::Function> convertToNgraph(const llvm::StringLiteral inputIR);
};

std::shared_ptr<ngraph::Function> ToNgraphBackendTests::convertToNgraph(const llvm::StringLiteral inputIR)
{
    ctx.getOrLoadDialect<IE::IEDialect>();
    auto module = mlir::parseSourceString(inputIR, &ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "");
    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    VPUX_THROW_UNLESS(func != nullptr, "");
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module.get(), netOp, netFunc);
    return IE::exportToNgraph(netOp, netFunc);
}

template <typename T>
std::shared_ptr<T> ToNgraphBackendTests::getNgraphNode(const llvm::StringLiteral inputIR)
{
    auto netGraph = convertToNgraph(inputIR);

    for (const auto& origNode : netGraph->get_ordered_ops()) {
        if (origNode->get_type_info() == T::get_type_info_static()) {
            return std::dynamic_pointer_cast<T>(origNode);
        }
    }
    VPUX_THROW("Op '{0}' missing in input IR", T::get_type_info_static());
}

//
// ConstDeclare
//
TEST_F(ToNgraphBackendTests, ConstDeclare_si64) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x3x16x16xsi64>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x3x1x1xsi64>
            }
            func @main(%arg0: tensor<1x3x16x16xsi64>) -> tensor<1x3x1x1xsi64> {
                %cst_0 = const.Declare tensor<1x3x1x1xsi64> = #const.Content<dense<[[[[-1000000000000000001]], [[-2]], [[3]]]]> : tensor<1x3x1x1xsi64>>
                return %cst_0 : tensor<1x3x1x1xsi64>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Constant>(inputIR);
    EXPECT_EQ(node->get_vector<long>(), std::vector<long>({-1000000000000000001,-2,3}));
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::i64);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,3,1,1}));
}

TEST_F(ToNgraphBackendTests, ConstDeclare_f16) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x3x16x16xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x3x1x1xf16>
            }
            func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x1x1xf16> {
                %cst_0 = const.Declare tensor<1x3x1x1xf16> = #const.Content<dense<[[[[-1.75976562e+00]], [[-1.98046875e+00]], [[-2.09960938e+00]]]]> : tensor<1x3x1x1xf16>>
                return %cst_0 : tensor<1x3x1x1xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Constant>(inputIR);
    EXPECT_EQ(node->cast_vector<float>(), std::vector<float>({-1.75976562e+00,-1.98046875e+00,-2.09960938e+00}));
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,3,1,1}));
}

//
// Convert
//
TEST_F(ToNgraphBackendTests, Convert) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<2x16xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<2x16xf16>
            }
            func @main(%arg0: tensor<2x16xf32>) -> tensor<2x16xf16> {
                %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
                return %0 : tensor<2x16xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Convert>(inputIR);
    EXPECT_EQ(node->get_destination_type(), ov::element::f16);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({2,16}));
}

//
// Softmax
//
TEST_F(ToNgraphBackendTests, Softmax) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x53xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x53xf16>
            }
            func @main(%arg0: tensor<1x53xf16>) -> tensor<1x53xf16> {
                %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x53xf16> -> tensor<1x53xf16>
                return %0 : tensor<1x53xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Softmax>(inputIR);
    EXPECT_EQ(node->get_axis(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,53}));
}

//
// Tile
//
TEST_F(ToNgraphBackendTests, Tile) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data0" : tensor<3x4x2xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<3x4x2xf32>
            }
            func @main(%arg0: tensor<3x4x2xf32>) -> tensor<3x4x2xf32> {
                %cst_0 = const.Declare tensor<3xsi64> = #const.Content<dense<[1,1,1]> : tensor<3xsi64>>
                %0 = IE.Tile(%arg0, %cst_0) : tensor<3x4x2xf32>, tensor<3xsi64> -> tensor<3x4x2xf32>
                return %0 : tensor<3x4x2xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Tile>(inputIR);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_partial_shape(0), ov::Shape({3,4,2}));
}

//
// Relu
//
TEST_F(ToNgraphBackendTests, Relu) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x4x2x3xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x4x2x3xf16>
            }
            func @main(%arg0: tensor<1x4x2x3xf16>) -> tensor<1x4x2x3xf16> {
                %0 = IE.ReLU(%arg0) : tensor<1x4x2x3xf16> -> tensor<1x4x2x3xf16>
                return %0 : tensor<1x4x2x3xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Relu>(inputIR);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,4,2,3}));
}

//
// Split
//
TEST_F(ToNgraphBackendTests, Split) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<2x6xf32>
            } outputsInfo :  {
                DataInfo "out1" : tensor<2x3xf32>
                DataInfo "out2" : tensor<2x3xf32>
            }
            func @main(%arg: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
                %0 = const.Declare tensor<si64> = #const.Content<dense<-1> : tensor<si64>>
                %1:2 = IE.Split(%arg, %0) {num_splits = 2} : tensor<2x6xf32>, tensor<si64> -> tensor<2x3xf32>, tensor<2x3xf32>
                return %1#0, %1#1 : tensor<2x3xf32>, tensor<2x3xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Split>(inputIR);
    EXPECT_EQ(node->get_num_splits(), 2);
    EXPECT_EQ(node->get_output_size(), 2);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_element_type(1), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({2,3}));
    EXPECT_EQ(node->get_output_shape(1), ov::Shape({2,3}));
}

//
// Power
//
TEST_F(ToNgraphBackendTests, Power) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x1x6xf32>
            }
            func @main(%arg0: tensor<8x1x6xf32>) -> tensor<8x1x6xf32> {
                %cst_0 = const.Declare tensor<1xf32> = #const.Content<dense<4.0> : tensor<1xf32>>
                %1 = IE.Power(%arg0, %cst_0) {auto_broadcast = "NUMPY"} : tensor<8x1x6xf32>, tensor<1xf32> -> tensor<8x1x6xf32>
                return %1 : tensor<8x1x6xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Power>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8, 1, 6}));
}

//
// Multiply
//
TEST_F(ToNgraphBackendTests, Multiply) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Multiply(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Multiply>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// Convolution
//
TEST_F(ToNgraphBackendTests, Convolution) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x3x16x16xf32>
                DataInfo "data2" : tensor<32x3x3x3xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x32x8x8xf32>
            }
            func @main(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<32x3x3x3xf32>) -> tensor<1x32x8x8xf32> {
                %1 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x16x16xf32>, tensor<32x3x3x3xf32> -> tensor<1x32x8x8xf32>
                return %1 : tensor<1x32x8x8xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Convolution>(inputIR);
    EXPECT_EQ(node->get_dilations(), ngraph::Strides({1, 1}));
    EXPECT_EQ(node->get_strides(), ngraph::Strides({2, 2}));
    EXPECT_EQ(node->get_pads_begin(), ngraph::CoordinateDiff({1, 1}));
    EXPECT_EQ(node->get_pads_end(), ngraph::CoordinateDiff({1, 1}));
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1, 32, 8, 8}));
}

//
// GroupConvolution
//
TEST_F(ToNgraphBackendTests, GroupConvolution) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x16x300x300xf32>
                DataInfo "data2" : tensor<16x1x1x3x3xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x16x300x300xf32>
            }
            func @main(%arg0: tensor<1x16x300x300xf32>, %arg1: tensor<16x1x1x3x3xf32>) -> tensor<1x16x300x300xf32> {
                %1 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x300x300xf32>, tensor<16x1x1x3x3xf32> -> tensor<1x16x300x300xf32>
                return %1 : tensor<1x16x300x300xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::GroupConvolution>(inputIR);
    EXPECT_EQ(node->get_dilations(), ngraph::Strides({1, 1}));
    EXPECT_EQ(node->get_strides(), ngraph::Strides({1, 1}));
    EXPECT_EQ(node->get_pads_begin(), ngraph::CoordinateDiff({1, 1}));
    EXPECT_EQ(node->get_pads_end(), ngraph::CoordinateDiff({1, 1}));
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,16,300,300}));
}

//
// Deconvolution
//
TEST_F(ToNgraphBackendTests, Deconvolution) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x64x64x112xf16>
                DataInfo "data2" : tensor<64x64x8x8xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x64x256x448xf16>
            }
            func @main(%arg0: tensor<1x64x64x112xf16>, %arg1: tensor<64x64x8x8xf16>) -> tensor<1x64x256x448xf16> {
                %1 = IE.Deconvolution(%arg0, %arg1) {dilations = [1, 1], output_padding = [0, 0], pads_begin = [2, 2], pads_end = [2, 2], strides = [4, 4]} : tensor<1x64x64x112xf16>, tensor<64x64x8x8xf16> -> tensor<1x64x256x448xf16>
                return %1 : tensor<1x64x256x448xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::ConvolutionBackpropData>(inputIR);
    EXPECT_EQ(node->get_dilations(), ngraph::Strides({1, 1}));
    EXPECT_EQ(node->get_output_padding(), ngraph::CoordinateDiff({0, 0}));
    EXPECT_EQ(node->get_pads_begin(), ngraph::CoordinateDiff({2, 2}));
    EXPECT_EQ(node->get_pads_end(), ngraph::CoordinateDiff({2, 2}));
    EXPECT_EQ(node->get_strides(), ngraph::Strides({4, 4}));
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_partial_shape(0), ov::Shape({1, 64, 256, 448}));
}

//
// AvgPool
//
TEST_F(ToNgraphBackendTests, Avgpool) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x128x56x56xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x128x28x28xf16>
            }
            func @main(%arg0: tensor<1x128x56x56xf16>) -> tensor<1x128x28x28xf16> {
                %1 = IE.AvgPool(%arg0) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "CEIL", strides = [2, 2]} : tensor<1x128x56x56xf16> -> tensor<1x128x28x28xf16>
                return %1 : tensor<1x128x28x28xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::AvgPool>(inputIR);
    EXPECT_TRUE(std::equal(node->get_kernel().begin(), node->get_kernel().end(), ov::Shape{2,2}.begin()));
    EXPECT_EQ(node->get_pads_begin(), ov::Shape({0, 0}));
    EXPECT_EQ(node->get_pads_end(), ov::Shape({0, 0}));
    EXPECT_EQ(node->get_rounding_type(), ov::op::RoundingType::CEIL);
    EXPECT_EQ(node->get_strides(), ngraph::Strides({2, 2}));
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1, 128, 28, 28}));
}

//
// MaxPool
//
TEST_F(ToNgraphBackendTests, Maxpool) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x128x56x56xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x128x28x28xf16>
            }
            func @main(%arg0: tensor<1x128x56x56xf16>) -> tensor<1x128x28x28xf16> {
                %1 = IE.MaxPool(%arg0) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "CEIL", strides = [2, 2]} : tensor<1x128x56x56xf16> -> tensor<1x128x28x28xf16>
                return %1 : tensor<1x128x28x28xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::MaxPool>(inputIR);
    EXPECT_TRUE(std::equal(node->get_kernel().begin(), node->get_kernel().end(), ov::Shape{2,2}.begin()));
    EXPECT_EQ(node->get_pads_begin(), ov::Shape({0, 0}));
    EXPECT_EQ(node->get_pads_end(), ov::Shape({0, 0}));
    EXPECT_EQ(node->get_rounding_type(), ov::op::RoundingType::CEIL);
    EXPECT_EQ(node->get_strides(), ngraph::Strides({2, 2}));
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1, 128, 28, 28}));
}

//
// Gather
//
TEST_F(ToNgraphBackendTests, Gather) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x2xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1xf16>
            }
            func @main(%arg0: tensor<1x2xf16>) -> tensor<1xf16> {
                %cst_0 = const.Declare tensor<si32> = #const.Content<dense<1> : tensor<si32>>
                %cst_1 = const.Declare tensor<si64> = #const.Content<dense<1> : tensor<si64>>
                %1 = IE.Gather(%arg0, %cst_0, %cst_1) {batch_dims = 0} : tensor<1x2xf16>, tensor<si32>, tensor<si64> -> tensor<1xf16>
                return %1 : tensor<1xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Gather>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1}));
}

//
// GatherElements
//
TEST_F(ToNgraphBackendTests, GatherElements) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<3x7x5xf16>
                DataInfo "data2" : tensor<3x10x5xsi32>
            } outputsInfo :  {
                DataInfo "out" : tensor<3x10x5xf16>
            }
            func @main(%arg0: tensor<3x7x5xf16>, %arg1: tensor<3x10x5xsi32>) -> tensor<3x10x5xf16> {
                %1 = IE.GatherElements(%arg0, %arg1) {axis = 1} : tensor<3x7x5xf16>, tensor<3x10x5xsi32> -> tensor<3x10x5xf16>
                return %1 : tensor<3x10x5xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::GatherElements>(inputIR);
    EXPECT_EQ(node->get_axis(), 1);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({3,10,5}));
}

//
// Clamp
//
TEST_F(ToNgraphBackendTests, Clamp) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x12xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x12xf16>
            }
            func @main(%arg0: tensor<1x12xf16>) -> tensor<1x12xf16> {
                %1 = IE.Clamp(%arg0) {max = 6.000000e+00 : f64, min = 0.000001e+00 : f64} : tensor<1x12xf16> -> tensor<1x12xf16>
                return %1 : tensor<1x12xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Clamp>(inputIR);
    EXPECT_EQ(node->get_min(), 0.000001e+00);
    EXPECT_EQ(node->get_max(), 6.000000e+00);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,12}));
}

//
// Elu
//
TEST_F(ToNgraphBackendTests, Elu) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x12xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x12xf16>
            }
            func @main(%arg0: tensor<1x12xf16>) -> tensor<1x12xf16> {
                %1 = IE.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x12xf16> -> tensor<1x12xf16>
                return %1 : tensor<1x12xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Elu>(inputIR);
    EXPECT_EQ(node->get_alpha(), 1.000000e+00);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,12}));
}

//
// Reshape
//
TEST_F(ToNgraphBackendTests, Reshape) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<96x1x3x3xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<96x1x1x3x3xf16>
            }
            func @main(%arg0: tensor<96x1x3x3xf16>) -> tensor<96x1x1x3x3xf16> {
                %cst_0 = const.Declare tensor<5xsi64> = #const.Content<dense<[96, 1, 1, 3, 3]> : tensor<5xsi64>>
                %1 = IE.Reshape(%arg0, %cst_0) {special_zero} : tensor<96x1x3x3xf16>, tensor<5xsi64> -> tensor<96x1x1x3x3xf16>
                return %1 : tensor<96x1x1x3x3xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Reshape>(inputIR);
    EXPECT_EQ(node->get_special_zero(), true);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({96,1,1,3,3}));
}

//
// Squeeze
//
TEST_F(ToNgraphBackendTests, Squeeze) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<4x4xf32> {
                %cst_0 = const.Declare tensor<2xsi64> = #const.Content<dense<[0, 1]> : tensor<2xsi64>>
                %0 = IE.Squeeze(%arg0, %cst_0) : tensor<1x1x4x4xf32>, tensor<2xsi64> -> tensor<4x4xf32>
                return %0 : tensor<4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Squeeze>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({4,4}));
}

//
// Sigmoid
//
TEST_F(ToNgraphBackendTests, Sigmoid) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Sigmoid(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Sigmoid>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// LRN
//
TEST_F(ToNgraphBackendTests, LRN) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x64x56x56xf16>
                DataInfo "data2" : tensor<1xsi64>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x64x56x56xf16>
            }
            func @main(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1xsi64>) -> tensor<1x64x56x56xf16> {
                %0 = IE.LRN(%arg0, %arg1) {alpha = 9.9999997473787516E-5 : f64, beta = 7.500000e-01 : f64, bias = 1.000000e+00 : f64, size = 5 : i64} : tensor<1x64x56x56xf16>, tensor<1xsi64> -> tensor<1x64x56x56xf16>
                return %0 : tensor<1x64x56x56xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::LRN>(inputIR);
    EXPECT_EQ(node->get_alpha(), 9.9999997473787516E-5);
    EXPECT_EQ(node->get_beta(), 7.500000e-01);
    EXPECT_EQ(node->get_bias(), 1.000000e+00);
    EXPECT_EQ(node->get_nsize(), 5);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,64,56,56}));
}

//
// LRN_IE
//
TEST_F(ToNgraphBackendTests, LRN_IE) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x64x56x56xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x64x56x56xf16>
            }
            func @main(%arg0: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
                %0 = IE.LRN_IE(%arg0) {alpha = 9.9999997473787516E-5 : f64, beta = 7.500000e-01 : f64, bias = 1.000000e+00 : f64, size = 5 : i64, region = "across"} : tensor<1x64x56x56xf16> -> tensor<1x64x56x56xf16>
                return %0 : tensor<1x64x56x56xf16>
            }
        }
    )";
    auto node = getNgraphNode<ngraph::op::LRN_IE>(inputIR);
    EXPECT_EQ(node->get_alpha(), 9.9999997473787516E-5);
    EXPECT_EQ(node->get_beta(), 7.500000e-01);
    EXPECT_EQ(node->get_bias(), 1.000000e+00);
    EXPECT_EQ(node->get_nsize(), 5);
    EXPECT_EQ(node->get_region(), "across");
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,64,56,56}));
}

//
// ReduceMax
//
TEST_F(ToNgraphBackendTests, ReduceMax) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<6x12x10x24xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<6x12xf16>
            }
            func @main(%arg0: tensor<6x12x10x24xf16>) -> tensor<6x12xf16> {
                %cst_0 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>
                %1 = IE.ReduceMax(%arg0, %cst_0) : tensor<6x12x10x24xf16>, tensor<2xsi64> -> tensor<6x12xf16>
                return %1 : tensor<6x12xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::ReduceMax>(inputIR);
    EXPECT_EQ(node->get_keep_dims(), false);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({6,12}));
}

//
// ReduceSum
//
TEST_F(ToNgraphBackendTests, ReduceSum) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x57x12x6xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x57xf16>
            }
            func @main(%arg0: tensor<1x57x12x6xf16>) -> tensor<1x57xf16> {
                %cst_0 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>
                %1 = IE.ReduceSum(%arg0, %cst_0) : tensor<1x57x12x6xf16>, tensor<2xsi64> -> tensor<1x57xf16>
                return %1 : tensor<1x57xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::ReduceSum>(inputIR);
    EXPECT_EQ(node->get_keep_dims(), false);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,57}));
}

//
// Unsqueeze
//
TEST_F(ToNgraphBackendTests, Unsqueeze) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<4x1x4x1xf32>
            }
            func @main(%arg0: tensor<4x4xf32>) -> tensor<4x1x4x1xf32> {
                %cst_0 = const.Declare tensor<2xsi64> = #const.Content<dense<[1, 3]> : tensor<2xsi64>>
                %0 = IE.Unsqueeze(%arg0, %cst_0) : tensor<4x4xf32>, tensor<2xsi64> -> tensor<4x1x4x1xf32>
                return %0 : tensor<4x1x4x1xf32>
            }
        }
    )";
    auto node = getNgraphNode<ngraph::op::Unsqueeze>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({4,1,4,1}));
}

//
// Minimum
//
TEST_F(ToNgraphBackendTests, Minimum) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Minimum(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Minimum>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// Maximum
//
TEST_F(ToNgraphBackendTests, Maximum) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Maximum(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Maximum>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// Add
//
TEST_F(ToNgraphBackendTests, Add) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Add>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// Divide
//
TEST_F(ToNgraphBackendTests, Divide) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Divide(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Divide>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// SquaredDifference
//
TEST_F(ToNgraphBackendTests, SquaredDifference) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.SquaredDiff(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::SquaredDifference>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// FloorMod
//
TEST_F(ToNgraphBackendTests, FloorMod) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.FloorMod(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::FloorMod>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// Proposal
//
TEST_F(ToNgraphBackendTests, Proposal) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @frozen_inference_graph  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x24x14x14xf32>
                DataInfo "data2" : tensor<1x48x14x14xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<10x5xf32>
            }
            func @main(%arg0: tensor<1x24x14x14xf32>, %arg1: tensor<1x48x14x14xf32>) -> tensor<10x5xf32> {
                %cst = const.Declare tensor<3xf32> = #const.Content<dense<[2.240000e+02, 2.240000e+02, 1.000000e+00]> : tensor<3xf32>>
                %0 = IE.Proposal(%arg0, %arg1, %cst) {proposal_attrs = {baseSize = 256 : i64, boxCoordinateScale = 1.000000e+01 : f64, boxSizeScale = 5.000000e+00 : f64, clipAfterNms = false, clipBeforeNms = true, featStride = 16 : i64, framework = "tensorflow", inferProbs = false, minSize = 1 : i64, nmsThresh = 0.69999998807907104 : f64, normalize = false, postNmsTopN = 10 : i64, preNmsTopN = 2147483647 : i64, ratio = [5.000000e-01, 1.000000e+00, 2.000000e+00], scale = [2.500000e-01, 5.000000e-01, 1.000000e+00, 2.000000e+00]}} : tensor<1x24x14x14xf32>, tensor<1x48x14x14xf32>, tensor<3xf32> -> tensor<10x5xf32>
                return %0 : tensor<10x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Proposal>(inputIR);
    const auto &attrs = node->get_attrs();
    EXPECT_EQ(attrs.base_size, 256);
    EXPECT_EQ(attrs.pre_nms_topn, 2147483647);
    EXPECT_EQ(attrs.post_nms_topn, 10);
    EXPECT_EQ(attrs.nms_thresh, 0.69999998807907104);
    EXPECT_EQ(attrs.feat_stride, 16);
    EXPECT_EQ(attrs.min_size, 1);
    EXPECT_EQ(attrs.ratio, std::vector<float>({5.000000e-01, 1.000000e+00, 2.000000e+00}));
    EXPECT_EQ(attrs.scale, std::vector<float>({2.500000e-01, 5.000000e-01, 1.000000e+00, 2.000000e+00}));
    EXPECT_EQ(attrs.clip_before_nms, true);
    EXPECT_EQ(attrs.clip_after_nms, false);
    EXPECT_EQ(attrs.normalize, false);
    EXPECT_EQ(attrs.box_size_scale, 5.000000e+00);
    EXPECT_EQ(attrs.box_coordinate_scale, 1.000000e+01);
    EXPECT_EQ(attrs.framework, "tensorflow");
    EXPECT_EQ(attrs.infer_probs, false);
    EXPECT_EQ(node->get_output_size(), 2);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({10,5}));
}

//
// FakeQuantize
//
TEST_F(ToNgraphBackendTests, FakeQuantize) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x128x8x8xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x128x8x8xf16>
            }
            func @main(%arg0: tensor<1x128x8x8xf16>) -> tensor<1x128x8x8xf16> {
                %cst_0 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f16>>
                %cst_1 = const.Declare tensor<f16> = #const.Content<dense<6.796880e+00> : tensor<f16>>
                %cst_2 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f16>>
                %cst_3 = const.Declare tensor<f16> = #const.Content<dense<6.796880e+00> : tensor<f16>>
                %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x128x8x8xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x128x8x8xf16>
                return %1 : tensor<1x128x8x8xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::FakeQuantize>(inputIR);
    EXPECT_EQ(node->get_auto_broadcast(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_levels(), 256);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,128,8,8}));
}

//
// MatMul
//
TEST_F(ToNgraphBackendTests, Matmul) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<5x10x15xf32>
                DataInfo "data2" : tensor<15x20xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<5x10x20xf32>
            }
            func @main(%arg0: tensor<5x10x15xf32>, %arg1: tensor<15x20xf32>) -> tensor<5x10x20xf32> {
                %1 = IE.MatMul(%arg0, %arg1) : tensor<5x10x15xf32>, tensor<15x20xf32> -> tensor<5x10x20xf32>
                return %1 : tensor<5x10x20xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::MatMul>(inputIR);
    EXPECT_EQ(node->get_transpose_a(), false);
    EXPECT_EQ(node->get_transpose_b(), false);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({5,10,20}));
}

//
// Tanh
//
TEST_F(ToNgraphBackendTests, Tanh) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Tanh(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Tanh>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Sqrt
//
TEST_F(ToNgraphBackendTests, Sqrt) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Sqrt(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Sqrt>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Log
//
TEST_F(ToNgraphBackendTests, Log) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Log(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Log>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Exp
//
TEST_F(ToNgraphBackendTests, Exp) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Exp(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Exp>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Hswish
//
TEST_F(ToNgraphBackendTests, Hswish) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.HSwish(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::HSwish>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Floor
//
TEST_F(ToNgraphBackendTests, Floor) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Floor(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Floor>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Round
//
TEST_F(ToNgraphBackendTests, Round) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Round(%arg0) {mode = "HALF_TO_EVEN"} : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Round>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Mish
//
TEST_F(ToNgraphBackendTests, Mish) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Mish(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Mish>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Erf
//
TEST_F(ToNgraphBackendTests, Erf) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x1x4x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x4x4xf32>
            }
            func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
                %0 = IE.Erf(%arg0) : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
                return %0 : tensor<1x1x4x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Erf>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,4}));
}

//
// Broadcast
//
TEST_F(ToNgraphBackendTests, Broadcast) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<16x1x1xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x16x50x50xf32>
            }
            func @main(%arg0: tensor<16x1x1xf32>) -> tensor<1x16x50x50xf32> {
                %cst_0 = const.Declare tensor<4xsi64> = #const.Content<dense<[1, 16, 50, 50]> : tensor<4xsi64>>
                %1 = IE.Broadcast(%arg0, %cst_0) {mode = "NUMPY"} : tensor<16x1x1xf32>, tensor<4xsi64> -> tensor<1x16x50x50xf32>
                return %1 : tensor<1x16x50x50xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Broadcast>(inputIR);
    EXPECT_EQ(node->get_broadcast_spec(), ov::op::BroadcastModeSpec{ov::op::BroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,16,50,50}));
}

//
// Transpose
//
TEST_F(ToNgraphBackendTests, Transpose) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x12x19x19xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x19x19x12xf16>
            }
            func @main(%arg0: tensor<1x12x19x19xf16>) -> tensor<1x19x19x12xf16> {
                %cst_0 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 2, 3, 1]> : tensor<4xsi64>>
                %0 = IE.Transpose(%arg0, %cst_0) : tensor<1x12x19x19xf16>, tensor<4xsi64> -> tensor<1x19x19x12xf16>
                return %0 : tensor<1x19x19x12xf16>
            }
        }
    )";
    auto node = getNgraphNode<ngraph::op::Transpose>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,19,19,12}));
}

//
// Interpolate
//
TEST_F(ToNgraphBackendTests, Interpolate) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x3x720x960xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x3x360x480xf16>
            }
            func @main(%arg0: tensor<1x3x720x960xf16>) -> tensor<1x3x360x480xf16> {
                %cst_1 = const.Declare tensor<2xsi64> = #const.Content<dense<[360, 480]> : tensor<2xsi64>>
                %cst_2 = const.Declare tensor<2xf32> = #const.Content<dense<5.000000e-01> : tensor<2xf32>>
                %cst_3 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>
                %1 = IE.Interpolate(%arg0, %cst_1, %cst_2, %cst_3) {attr = {antialias = false, coord_mode = "align_corners", cube_coeff = -7.500000e-01 : f64, mode = "linear_onnx", nearest_mode = "simple", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"}, operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x3x720x960xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x360x480xf16>
                return %1 : tensor<1x3x360x480xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Interpolate>(inputIR);
    auto attrs = node->get_attrs();
    EXPECT_EQ(attrs.antialias, false);
    EXPECT_EQ(attrs.coordinate_transformation_mode, opset_latest::Interpolate::CoordinateTransformMode::ALIGN_CORNERS);
    EXPECT_EQ(attrs.cube_coeff, -7.5e-01);
    EXPECT_EQ(attrs.mode, opset_latest::Interpolate::InterpolateMode::LINEAR_ONNX);
    EXPECT_EQ(attrs.nearest_mode, opset_latest::Interpolate::NearestMode::SIMPLE);
    EXPECT_EQ(attrs.pads_begin, InferenceEngine::SizeVector({0, 0, 0, 0}));
    EXPECT_EQ(attrs.pads_end, InferenceEngine::SizeVector({0, 0, 0, 0}));
    EXPECT_EQ(attrs.shape_calculation_mode, opset_latest::Interpolate::ShapeCalcMode::SIZES);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,3,360,480}));
}

//
// TopK
//
TEST_F(ToNgraphBackendTests, Topk) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x12x720x960xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x720x960xf16>
            }
            func @main(%arg0: tensor<1x12x720x960xf16>) -> tensor<1x1x720x960xf16> {
                %cst_0 = const.Declare tensor<si32> = #const.Content<dense<1> : tensor<si32>>
                %output_values, %target_shape = IE.TopK(%arg0, %cst_0) {axis = 1 : i64, element_type = si64, mode = "MAX", sort = "SORT_INDICES"} : tensor<1x12x720x960xf16>, tensor<si32> -> tensor<1x1x720x960xf16>, tensor<1x1x720x960xsi64>
                return %output_values : tensor<1x1x720x960xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::TopK>(inputIR);
    EXPECT_EQ(node->get_axis(), 1);
    EXPECT_EQ(node->get_index_element_type(), ngraph::element::i64);
    EXPECT_EQ(node->get_mode(), ngraph::op::TopKMode::MAX);
    EXPECT_EQ(node->get_sort_type(), ngraph::op::TopKSortType::SORT_INDICES);
    EXPECT_EQ(node->get_output_size(), 2);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_element_type(1), ov::element::i64);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,720,960}));
    EXPECT_EQ(node->get_output_shape(1), ov::Shape({1,1,720,960}));
}

//
// RegionYolo
//
TEST_F(ToNgraphBackendTests, RegionYolo) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x255x38x38xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x255x38x38xf16>
            }
            func @main(%arg0: tensor<1x255x38x38xf16>) -> tensor<1x255x38x38xf16> {
                %0 = IE.RegionYolo(%arg0) {anchors = [1.0e+01, 1.3e+01, 1.6e+01, 3.0e+01, 3.3e+01, 2.3e+01, 3.0e+01, 6.1e+01, 6.2e+01, 4.5e+01, 5.9e+01, 1.19e+02, 1.16e+02, 9.0e+01, 1.56e+02, 1.98e+02, 3.73e+02, 3.26e+02], axis = 1 : i64, classes = 80 : i64, coords = 4 : i64, do_softmax = false, end_axis = 3 : i64, mask = [3, 4, 5], regions = 9 : i64} : tensor<1x255x38x38xf16> -> tensor<1x255x38x38xf16>
                return %0 : tensor<1x255x38x38xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::RegionYolo>(inputIR);
    EXPECT_EQ(node->get_anchors(),
        std::vector<float>({1.0e+01, 1.3e+01, 1.6e+01, 3.0e+01, 3.3e+01, 2.3e+01, 3.0e+01, 6.1e+01, 6.2e+01, 4.5e+01, 5.9e+01, 1.19e+02, 1.16e+02, 9.0e+01, 1.56e+02, 1.98e+02, 3.73e+02, 3.26e+02}));
    EXPECT_EQ(node->get_axis(), 1);
    EXPECT_EQ(node->get_num_classes(), 80);
    EXPECT_EQ(node->get_num_coords(), 4);
    EXPECT_EQ(node->get_do_softmax(), false);
    EXPECT_EQ(node->get_end_axis(), 3);
    EXPECT_EQ(node->get_mask(), std::vector<int64_t>({3, 4, 5}));
    EXPECT_EQ(node->get_num_regions(), 9);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,255,38,38}));
}

//
// ReorgYolo
//
TEST_F(ToNgraphBackendTests, ReorgYolo) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x64x26x26xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x256x13x13xf16>
            }
            func @main(%arg0: tensor<1x64x26x26xf16>) -> tensor<1x256x13x13xf16> {
                %0 = IE.ReorgYolo(%arg0) {stride = 2 : i64} : tensor<1x64x26x26xf16> -> tensor<1x256x13x13xf16>
                return %0 : tensor<1x256x13x13xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::ReorgYolo>(inputIR);
    EXPECT_EQ(node->get_strides().front(), 2);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,256,13,13}));
}

//
// DetectionOutput
//
TEST_F(ToNgraphBackendTests, DetectionOutput) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x32640xf16>
                DataInfo "data2" : tensor<1x16320xf16>
                DataInfo "data3" : tensor<1x2x32640xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x200x7xf16>
            }
            func @main(%arg0: tensor<1x32640xf16>, %arg1: tensor<1x16320xf16>, %arg2: tensor<1x2x32640xf32>) -> tensor<1x1x200x7xf16> {
                %1 = IE.DetectionOutput(%arg0, %arg1, %arg2) {attr = {background_label_id = 0 : i64, clip_after_nms = false, clip_before_nms = false, code_type = "caffe.PriorBoxParameter.CENTER_SIZE", confidence_threshold = 0.0099999997764825821 : f64, decrease_label_id = false, input_height = 1 : i64, input_width = 1 : i64, keep_top_k = [200], nms_threshold = 0.44999998807907104 : f64, normalized = true, num_classes = 2 : i64, objectness_score = 0.000000e+00 : f64, share_location = true, top_k = 400 : i64, variance_encoded_in_target = false}, operand_segment_sizes = dense<[1, 1, 1, 0, 0]> : vector<5xi32>} : tensor<1x32640xf16>, tensor<1x16320xf16>, tensor<1x2x32640xf32> -> tensor<1x1x200x7xf16>
                return %1 : tensor<1x1x200x7xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::DetectionOutput>(inputIR);
    EXPECT_EQ(node->get_attrs().background_label_id, 0);
    EXPECT_EQ(node->get_attrs().clip_after_nms, false);
    EXPECT_EQ(node->get_attrs().clip_before_nms, false);
    EXPECT_EQ(node->get_attrs().code_type, "caffe.PriorBoxParameter.CENTER_SIZE");
    EXPECT_EQ(node->get_attrs().confidence_threshold, 0.0099999997764825821);
    EXPECT_EQ(node->get_attrs().decrease_label_id, false);
    EXPECT_EQ(node->get_attrs().input_height, 1);
    EXPECT_EQ(node->get_attrs().input_width, 1);
    EXPECT_EQ(node->get_attrs().keep_top_k, std::vector<int32_t>{200});
    EXPECT_EQ(node->get_attrs().nms_threshold, 0.44999998807907104);
    EXPECT_EQ(node->get_attrs().normalized, true);
    EXPECT_EQ(node->get_attrs().num_classes, 2);
    EXPECT_EQ(node->get_attrs().objectness_score, 0.0);
    EXPECT_EQ(node->get_attrs().share_location, true);
    EXPECT_EQ(node->get_attrs().top_k, 400);
    EXPECT_EQ(node->get_attrs().variance_encoded_in_target, false);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1, 1, 200, 7}));
}

//
// NormalizeIE
//
TEST_F(ToNgraphBackendTests, NormalizeIE) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x128x25x43xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x128x25x43xf16>
            }
            func @main(%arg0: tensor<1x128x25x43xf16>) -> tensor<1x128x25x43xf16> {
                %cst_0 = const.Declare tensor<1xf16> = #const.Content<dense<1.000000e+00> : tensor<1xf16>>
                %0 = IE.NormalizeIE(%arg0, %cst_0) {across_spatial = false, channel_shared = true, eps = 9.9999999392252903E-9 : f64} : tensor<1x128x25x43xf16>, tensor<1xf16> -> tensor<1x128x25x43xf16>
                return %0 : tensor<1x128x25x43xf16>
            }
        }
    )";
    auto node = getNgraphNode<ngraph::op::NormalizeIE>(inputIR);
    EXPECT_EQ(node->get_across_spatial(), false);
    EXPECT_EQ(node->get_channel_shared(), true);
    EXPECT_EQ(node->get_eps(), 9.9999999392252903E-9);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,128,25,43}));
}

//
// MVN
//
TEST_F(ToNgraphBackendTests, MVN) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x48x12x20xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x48x12x20xf16>
            }
            func @main(%arg0: tensor<1x48x12x20xf16>) -> tensor<1x48x12x20xf16> {
                %0 = IE.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x48x12x20xf16> -> tensor<1x48x12x20xf16>
                return %0 : tensor<1x48x12x20xf16>
            }
        }
    )";
    auto node = getNgraphNode<ngraph::opset4::MVN>(inputIR);
    EXPECT_EQ(node->get_across_channels(), false);
    EXPECT_EQ(node->get_eps(), 9.9999997473787516E-6);
    EXPECT_EQ(node->get_normalize_variance(), true);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,48,12,20}));
}

//
// Concat
//
TEST_F(ToNgraphBackendTests, Concat) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x2x3x4xf32>
                DataInfo "data2" : tensor<1x2x3x4xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x4x3x4xf32>
            }
            func @main(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
                %0 = IE.Concat(%arg0, %arg1) {per_axis = {axis = 1 : i64}} : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
                return %0 : tensor<1x4x3x4xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Concat>(inputIR);
    EXPECT_EQ(node->get_axis(), 1);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,4,3,4}));
}

//
// ROIPooling
//
TEST_F(ToNgraphBackendTests, ROIPooling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x576x14x14xf32>
                DataInfo "data2" : tensor<10x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<10x576x14x14xf32>
            }
            func @main(%arg0: tensor<1x576x14x14xf32>, %arg1: tensor<10x5xf32>) -> tensor<10x576x14x14xf32> {
                %0 = IE.ROIPooling(%arg0, %arg1) {output_size = [14, 14], spatial_scale = 1.0 : f64, method = "bilinear"} : tensor<1x576x14x14xf32>, tensor<10x5xf32> -> tensor<10x576x14x14xf32>
                return %0 : tensor<10x576x14x14xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::ROIPooling>(inputIR);
    EXPECT_EQ(node->get_output_size(), ov::Shape({14,14}));
    EXPECT_EQ(node->get_spatial_scale(), 1.0);
    EXPECT_EQ(node->get_method(), "bilinear");
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({10,576,14,14}));
}

//
// StridedSlice
//
TEST_F(ToNgraphBackendTests, StridedSlice) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x80x28x28xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x76x28x28xf16>
            }
            func @main(%arg0: tensor<1x80x28x28xf16>) -> tensor<1x76x28x28xf16> {
                %cst_0 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, -76, 0, 0]> : tensor<4xsi64>>
                %cst_1 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 2147483647, 0, 0]> : tensor<4xsi64>>
                %cst_2 = const.Declare tensor<4xsi64> = #const.Content<dense<1> : tensor<4xsi64>>
                %0 = IE.StridedSlice(%arg0, %cst_0, %cst_1, %cst_2) {begin_mask = [1, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [1, 0, 1, 1], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<1> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0]} : tensor<1x80x28x28xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x76x28x28xf16>
                return %0 : tensor<1x76x28x28xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::StridedSlice>(inputIR);
    EXPECT_EQ(node->get_begin_mask(), std::vector<int64_t>({1, 0, 1, 1}));
    EXPECT_EQ(node->get_ellipsis_mask(), std::vector<int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(node->get_end_mask(), std::vector<int64_t>({1, 0, 1, 1}));
    EXPECT_EQ(node->get_new_axis_mask(), std::vector<int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(node->get_shrink_axis_mask(), std::vector<int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,76,28,28}));
}

//
// PRelu
//
TEST_F(ToNgraphBackendTests, PRelu) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x64x208x208xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x64x208x208xf16>
            }
            func @main(%arg0: tensor<1x64x208x208xf16>) -> tensor<1x64x208x208xf16> {
                %cst_0 = const.Declare tensor<1xf32> = #const.Content<dense<1.000000e-01> : tensor<1xf32>>
                %0 =  IE.PRelu(%arg0, %cst_0) : tensor<1x64x208x208xf16>, tensor<1xf32> -> tensor<1x64x208x208xf16>
                return %0 : tensor<1x64x208x208xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::PRelu>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,64,208,208}));
}

//
// Swish
//
TEST_F(ToNgraphBackendTests, Swish) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x16x300x300xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x16x300x300xf32>
            }
            func @main(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
                %1 = IE.Swish(%arg0) {beta_value = 1.000000e+00} : tensor<1x16x300x300xf32> -> tensor<1x16x300x300xf32>
                return %1 : tensor<1x16x300x300xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Swish>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,16,300,300}));
}

//
// GRN
//
TEST_F(ToNgraphBackendTests, GRN) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x48x12x20xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x48x12x20xf16>
            }
            func @main(%arg0: tensor<1x48x12x20xf16>) -> tensor<1x48x12x20xf16> {
                %0 = IE.GRN(%arg0) {bias = 1.0} : tensor<1x48x12x20xf16> -> tensor<1x48x12x20xf16>
                return %0 : tensor<1x48x12x20xf16>
            }
        }
    )";
    auto node = getNgraphNode<ngraph::opset4::GRN>(inputIR);
    EXPECT_EQ(node->get_bias(), 1.0);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,48,12,20}));
}

//
// Negative
//
TEST_F(ToNgraphBackendTests, Negative) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x48x12x20xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x48x12x20xf16>
            }
            func @main(%arg0: tensor<1x48x12x20xf16>) -> tensor<1x48x12x20xf16> {
                %0 = IE.Negative(%arg0) : tensor<1x48x12x20xf16> -> tensor<1x48x12x20xf16>
                return %0 : tensor<1x48x12x20xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Negative>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,48,12,20}));
}

//
// CTCGreedyDecoder
//
TEST_F(ToNgraphBackendTests, CTCGreedyDecoder) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<88x1x71xf16>
                DataInfo "data2" : tensor<88x1xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x88x1x1xf16>
            }
            func @main(%arg0: tensor<88x1x71xf16>, %arg1: tensor<88x1xf16>) -> tensor<1x88x1x1xf16> {
                %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<88x1x71xf16>, tensor<88x1xf16> -> tensor<1x88x1x1xf16>
                return %0 : tensor<1x88x1x1xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::CTCGreedyDecoder>(inputIR);
    EXPECT_EQ(node->get_ctc_merge_repeated(), true);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,88,1,1}));
}

//
// CTCGreedyDecoderSeqLen
//
TEST_F(ToNgraphBackendTests, CTCGreedyDecoderSeqLen) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x20x128xf16>
                DataInfo "data2" : tensor<8xsi32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x20xsi32>
            }
            func @main(%arg0: tensor<8x20x128xf16>, %arg1: tensor<8xsi32>) -> tensor<8x20xsi32> {
                %cst_0 = const.Declare tensor<1xsi32> = #const.Content<dense<70> : tensor<1xsi32>>
                %1, %2 = IE.CTCGreedyDecoderSeqLen(%arg0, %arg1, %cst_0) {mergeRepeated} : tensor<8x20x128xf16>, tensor<8xsi32>, tensor<1xsi32> -> tensor<8x20xsi32>, tensor<8xsi32>
                return %1 : tensor<8x20xsi32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::CTCGreedyDecoderSeqLen>(inputIR);
    EXPECT_EQ(node->get_merge_repeated(), true);
    EXPECT_EQ(node->get_output_size(), 2);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::i32);
    EXPECT_EQ(node->get_output_element_type(1), ov::element::i32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,20}));
    EXPECT_EQ(node->get_output_shape(1), ov::Shape({8}));
}

//
// Pad
//
TEST_F(ToNgraphBackendTests, Pad) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x5x10x11xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x11x12x12xf16>
            }
            func @main(%arg0: tensor<1x5x10x11xf16>) -> tensor<1x11x12x12xf16> {
                %0 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 3, 0, 1]> : tensor<4xsi64>>
                %1 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 3, 2, 0]> : tensor<4xsi64>>
                %2 = const.Declare tensor<f16> = #const.Content<dense<1.000000e+00> : tensor<f16>>
                %3 = IE.Pad(%arg0)[%0, %1, %2] {mode = "SYMMETRIC"} : tensor<1x5x10x11xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<f16> -> tensor<1x11x12x12xf16>
                return %3 : tensor<1x11x12x12xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Pad>(inputIR);
    EXPECT_EQ(node->get_pad_mode(), ov::op::PadMode::SYMMETRIC);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,11,12,12}));
}

//
// LSTMCell
//
TEST_F(ToNgraphBackendTests, LSTMCell) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x128xf16>
                DataInfo "data2" : tensor<1x128xf16>
                DataInfo "data3" : tensor<1x128xf16>
                DataInfo "data4" : tensor<512x128xf16>
                DataInfo "data5" : tensor<512x128xf16>
                DataInfo "data6" : tensor<512xf16>
            } outputsInfo :  {
                DataInfo "out1" : tensor<1x128xf16>
            }
            func @main(%arg0: tensor<1x128xf16>, %arg1: tensor<1x128xf16>, %arg2: tensor<1x128xf16>, %arg3: tensor<512x128xf16>, %arg4: tensor<512x128xf16>, %arg5: tensor<512xf16>) -> tensor<1x128xf16> {
                %0, %1 = IE.LSTMCell(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {hiddenSize = 128} : tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<512x128xf16>, tensor<512x128xf16>, tensor<512xf16> -> tensor<1x128xf16>, tensor<1x128xf16>
                return %0 : tensor<1x128xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::LSTMCell>(inputIR);
    EXPECT_EQ(node->get_output_size(), 2);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_element_type(1), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,128}));
    EXPECT_EQ(node->get_output_shape(1), ov::Shape({1,128}));
}

//
// Subtract
//
TEST_F(ToNgraphBackendTests, Subtract) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Subtract(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Subtract>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// LogicalAnd
//
TEST_F(ToNgraphBackendTests, And) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xi8>
                DataInfo "data2" : tensor<7x1x5xi8>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xi8>
            }
            func @main(%arg0: tensor<8x1x6x1xi8>, %arg1: tensor<7x1x5xi8>) -> tensor<8x7x6x5xi8> {
                %1 = IE.And(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xi8>, tensor<7x1x5xi8> -> tensor<8x7x6x5xi8>
                return %1 : tensor<8x7x6x5xi8>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::LogicalAnd>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::boolean);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// LSTMSequence
// 
TEST_F(ToNgraphBackendTests, LSTMSequence) {
        constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "image_tensor1" : tensor<1x4x16xf32>
                DataInfo "image_tensor2" : tensor<1x1x128xf32>
                DataInfo "image_tensor3" : tensor<1x1x128xf32>
                DataInfo "image_tensor5" : tensor<1x512x16xf32>
                DataInfo "image_tensor6" : tensor<1x512x128xf32>
                DataInfo "image_tensor7" : tensor<1x512xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x1x128xf32>
            }
            func @main(%arg0: tensor<1x4x16xf32>, %arg1: tensor<1x1x128xf32>, %arg2: tensor<1x1x128xf32>, %arg4: tensor<1x512x16xf32>, %arg5: tensor<1x512x128xf32>, %arg6: tensor<1x512xf32>) -> tensor<1x1x128xf32> {
                %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %arg4, %arg5, %arg6) {direction = "FORWARD", sequenceLength = 224 : i64} : tensor<1x4x16xf32>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x512x16xf32>, tensor<1x512x128xf32>, tensor<1x512xf32> -> tensor<1x1x224x128xf32>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
                return %outputHiddenState : tensor<1x1x128xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::LSTMSequence>(inputIR);
    EXPECT_EQ(node->get_direction(), ngraph::op::RecurrentSequenceDirection::FORWARD);
    EXPECT_EQ(node->get_output_size(), 3);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,1,4,128}));
}

//
// Ceiling
//
TEST_F(ToNgraphBackendTests, Ceiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x48x12x20xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x48x12x20xf16>
            }
            func @main(%arg0: tensor<1x48x12x20xf16>) -> tensor<1x48x12x20xf16> {
                %0 = IE.Ceiling(%arg0) : tensor<1x48x12x20xf16> -> tensor<1x48x12x20xf16>
                return %0 : tensor<1x48x12x20xf16>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Ceiling>(inputIR);
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,48,12,20}));
}

//
// Equal
//
TEST_F(ToNgraphBackendTests, Equal) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Equal(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Equal>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::boolean);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// Less
//
TEST_F(ToNgraphBackendTests, Less) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.Less(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::Less>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::boolean);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

//
// LessEqual
//
TEST_F(ToNgraphBackendTests, LessEqual) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<8x1x6x1xf32>
                DataInfo "data2" : tensor<7x1x5xf32>
            } outputsInfo :  {
                DataInfo "out" : tensor<8x7x6x5xf32>
            }
            func @main(%arg0: tensor<8x1x6x1xf32>, %arg1: tensor<7x1x5xf32>) -> tensor<8x7x6x5xf32> {
                %1 = IE.LessEqual(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<8x1x6x1xf32>, tensor<7x1x5xf32> -> tensor<8x7x6x5xf32>
                return %1 : tensor<8x7x6x5xf32>
            }
        }
    )";
    auto node = getNgraphNode<opset_latest::LessEqual>(inputIR);
    EXPECT_EQ(node->get_autob(), ov::op::AutoBroadcastSpec{ov::op::AutoBroadcastType::NUMPY});
    EXPECT_EQ(node->get_output_size(), 1);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::boolean);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({8,7,6,5}));
}

TEST_F(ToNgraphBackendTests, MultipleOps1) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x4x16x16xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x2x1x1xf32>
                DataInfo "out2" : tensor<1x2x1x1xf32>
            }
            func @main(%arg0: tensor<1x4x16x16xf16>) -> (tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>) {
                %cst_0 = const.Declare tensor<1x4x1x1xf16> = #const.Content<dense<[[[[-1.75976562e+00]], [[-1.98046875e+00]], [[-2.09960938e+00]], [[-2.35624e+00]]]]> : tensor<1x4x1x1xf16>>
                %0 = IE.Convert(%cst_0) {dstElemType = f32} : tensor<1x4x1x1xf16> -> tensor<1x4x1x1xf32>
                %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x4x1x1xf32> -> tensor<1x4x1x1xf32>
                %cst_1 = const.Declare tensor<4xsi64> = #const.Content<dense<[1,1,1,1]> : tensor<4xsi64>>
                %2 = IE.Tile(%1, %cst_1) : tensor<1x4x1x1xf32>, tensor<4xsi64> -> tensor<1x4x1x1xf32>
                %3 = IE.ReLU(%2) : tensor<1x4x1x1xf32> -> tensor<1x4x1x1xf32>
                %cst_2 = const.Declare tensor<si64> = #const.Content<dense<1> : tensor<si64>>
                %4:2 = IE.Split(%3, %cst_2) {num_splits = 2} : tensor<1x4x1x1xf32>, tensor<si64> -> tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>
                return %4#0, %4#1 : tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>
            }
        }
    )";
    getNgraphNode<opset_latest::Constant>(inputIR);
    getNgraphNode<opset_latest::Convert>(inputIR);
    getNgraphNode<opset_latest::Softmax>(inputIR);
    getNgraphNode<opset_latest::Tile>(inputIR);
    getNgraphNode<opset_latest::Relu>(inputIR);
    getNgraphNode<opset_latest::Split>(inputIR);
    auto node = getNgraphNode<opset_latest::Result>(inputIR);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,2,1,1}));
}

TEST_F(ToNgraphBackendTests, MultipleOps2) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test  {
            IE.CNNNetwork entryPoint : @main inputsInfo :  {
                DataInfo "data1" : tensor<1x16x128x64xf16>
            } outputsInfo :  {
                DataInfo "out" : tensor<1x16x1x1xf16>
            }
            func @main(%arg0: tensor<1x16x128x64xf16>) -> tensor<1x16x1x1xf16> {
                %1 = IE.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x16x128x64xf16> -> tensor<1x16x128x64xf16>
                %cst_1 = const.Declare tensor<1x16x1x1xf16> = #const.Content<dense<[[[[3.488770e-01]], [[3.815920e-01]], [[3.688960e-01]], [[4.204100e-01]], [[4.765630e-01]], [[4.873050e-01]], [[2.734380e-01]], [[3.015140e-01]], [[3.559570e-01]], [[2.941890e-01]], [[6.357420e-01]], [[4.226070e-01]], [[4.562990e-01]], [[2.343750e-01]], [[3.713380e-01]], [[2.338870e-01]]]]> : tensor<1x16x1x1xf16>>
                %2 = IE.Multiply(%1, %cst_1) {auto_broadcast = "NUMPY"} : tensor<1x16x128x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x128x64xf16>
                %cst_2 = const.Declare tensor<1x16x1x1xf16> = #const.Content<dense<[[[[4.565430e-01]], [[8.754880e-01]], [[6.494140e-01]], [[9.194330e-01]], [[2.106930e-01]], [[2.159420e-01]], [[7.163080e-01]], [[9.399410e-01]], [[7.661130e-01]], [[8.901360e-01]], [[2.349850e-01]], [[1.948240e-01]], [[1.062500e+00]], [[8.740230e-01]], [[4.272460e-01]], [[9.604490e-01]]]]> : tensor<1x16x1x1xf16>>
                %3 = IE.Add(%2, %cst_2) {auto_broadcast = "NUMPY"} : tensor<1x16x128x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x128x64xf16>
                %4 = IE.MaxPool(%3) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = "FLOOR", strides = [2, 2]} : tensor<1x16x128x64xf16> -> tensor<1x16x64x32xf16>
                %5 = IE.ReLU(%4) : tensor<1x16x64x32xf16> -> tensor<1x16x64x32xf16>
                %cst_3 = const.Declare tensor<16x16x1x1xf16> = #const.Content<dense<"0x8F2ABE2C2EA2343148A45C30A52ACFA5362CF4276FAE4E2EC49CDB2C7DB0132C6AAEE0136826FA36B2AF1AB3151D572C94AF62A9932060A963AC92252521122AF9294532EE2E0CABAC2233AEFFA61EA90F270323122D783413AD16287130BB2C6BB49B2B012FF0AFBB250CA29329BF1F41A958279A2D51B4A2A36CB5E5B6ADAC48A5C9AC0EACA69C1030FFA2B3AA122D5C298DB1BA33A8A8CBAE8B9FDF30C634032455B496B081A2BA2963A4E22EA529672D67A0323077AA8831E4AF08B9C82900800000000000800080008000000080008000800000008000000080000000008AA28B32A02F18AD5D2D4A2AB6A402AA28ADF3A8679BC62EC3A9CE33E0346F2B632E40A63F1F52B59EAE5035F8A4A7A211A4532A2DABC5A16FA6C0AAA1991D29C52990AAF7231BADCF22509D51AF84AD8528A9B26B2DA02D4CAC9C2B292A302CF52EC2B244B040ADD92D0DAC97AB2629A5AEFDA05A34102A2FB05A302EAC3430E8A3AB31DDB25AB632ADE3B5492728A4A71A23AA9D25F227C4B05C21F12F5C290330EC2F0AB1CCA38DA9742AEA2B0A2941AC3CAE48B3862E6D313C34023611ABEDAD0432972E0428982857AA61324BA9759F9A20F7A0EFB048A9E1B5CDAD0EAD522F442ED72D742CAD24FD2BACAC3A2E5C1C99A8B5B34632469FD62DCC35B92A5CAAECA821A69BA3FFA637A9102E82212AAC7DACEFAD07B07DAAE7B1D9B925AD"> : tensor<16x16x1x1xf16>>
                %6 = IE.Convolution(%5, %cst_3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x32xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x64x32xf16>
                %cst_4 = const.Declare tensor<16x1x1x3x3xf16> = #const.Content<dense<"0x393A2140613CC8BC87C264BE52B09439CB3F4E25C5B1A2193336D33DE938B79F8D29D0ABD1B111BC0FBACBBAFABA29BA8AB684305BB123B94EB5AABEE8BA023C34BBA1B8AA3550BBEE3899307E364E2FBFBE9E2F693647314C3823B706BB92B8062FED311F309E3386364A34008000800080008000000080008000000080ED3AD0AF49B73F4053BB4ABD413C2EBCB3ACA1B308ABD03100B8C39D273BAAB72AB4F237ACB99BB1A3343FBCEE340E350DB723359B3A103666B9DB3877B21BC047B436366EAB4D3353AD7BB545B0F8B1B1B909B0452813BA34AC97B1DEBA3FB005B48CB117B39C36D43DD53753BD31BB69C2F0BF0EC2C5C17B4268389C438B9656B99EA2DBB8E6B51EB501B0A8B7FCAAB72B382CAC2EAAB48ABD19B8542DBDB5E831"> : tensor<16x1x1x3x3xf16>>
                %7 = IE.GroupConvolution(%6, %cst_4) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x64x32xf16>, tensor<16x1x1x3x3xf16> -> tensor<1x16x64x32xf16>
                %8 = IE.AvgPool(%7) {exclude_pads, kernel_size = [64, 32], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x16x64x32xf16> -> tensor<1x16x1x1xf16>
                %9 = IE.Sigmoid(%8) : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
                return %9 : tensor<1x16x1x1xf16>
            }
        }
    )";
    getNgraphNode<ngraph::opset4::MVN>(inputIR);
    getNgraphNode<opset_latest::Constant>(inputIR);
    getNgraphNode<opset_latest::Multiply>(inputIR);
    getNgraphNode<opset_latest::Add>(inputIR);
    getNgraphNode<opset_latest::MaxPool>(inputIR);
    getNgraphNode<opset_latest::Relu>(inputIR);
    getNgraphNode<opset_latest::Convolution>(inputIR);
    getNgraphNode<opset_latest::GroupConvolution>(inputIR);
    getNgraphNode<opset_latest::AvgPool>(inputIR);
    getNgraphNode<opset_latest::Sigmoid>(inputIR);
    auto node = getNgraphNode<opset_latest::Result>(inputIR);
    EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(node->get_output_shape(0), ov::Shape({1,16,1,1}));
}
