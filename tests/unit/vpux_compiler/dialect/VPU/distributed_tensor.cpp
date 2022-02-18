//
// Copyright 2022 Intel Corporation
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

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}


TEST(MLIR_NDTypeInterface, DistributedTensorType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        !InputDistributed = type !VPU.DistributedTensor<
            1x32x16x16xf16, #NHWC, @CMX_NN, {
            mode = OVERLAPPED,
            num_tiles = [1, 1, 4, 1],
            kernel = [3, 3],
            pads = {bottom = 1, left = 1, right = 1, top = 1}
        }>

        !WeightsDistributed = type !VPU.DistributedTensor<
            64x32x3x3xf16, #NHWC, @CMX_NN, {
            mode = DUPLICATED
        }>

        !OutputDistributed = type !VPU.DistributedTensor<
            1x64x16x16xf16, #NHWC, @CMX_NN, {
            mode = SEGMENTED,
            num_tiles = [1, 1, 4, 1]
        }>

        !InputStub_CMX = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        !WeightsStub_CMX = type tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        !OutputStub_CMX = type tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        module @test {
            func @main(%arg_input: !InputDistributed, %arg_wt: !WeightsDistributed) -> !OutputDistributed {
                %output_cmx = VPU.NCE.ClusterTiling (
                        %arg_input as %arg1: !InputStub_CMX,
                        %arg_wt as %arg2: !WeightsStub_CMX) -> !OutputDistributed {
                    %0 = VPU.NCE.Convolution(%arg1, %arg2) (bias : #const.Content<dense<1.000000e+00> : tensor<1x64x1x1xf16>>) {
                            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                            strides = [1, 1]
                        } -> !OutputStub_CMX
                    VPU.Yield %0
                }
                return %output_cmx: !OutputDistributed
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<vpux::VPU::NCEClusterTilingOp>(op)) {
            // Distributed "segmented" output type
            const auto ndOutputType = taskOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();

            EXPECT_TRUE(ndOutputType != nullptr) << "Output is not of vpux::NDTypeInterface type";

            EXPECT_EQ(ndOutputType.getShape(), vpux::ShapeRef({1, 64, 16, 16}));
            EXPECT_EQ(ndOutputType.getMemShape(), vpux::MemShape({1, 16, 16, 64}));

            EXPECT_TRUE(ndOutputType.hasRank());
            EXPECT_EQ(ndOutputType.getRank(), 4);
            EXPECT_EQ(ndOutputType.getNumElements(), 64*16*16);

            EXPECT_TRUE(ndOutputType.getElementType().isa<mlir::Float16Type>());

            EXPECT_EQ(ndOutputType.getDimsOrder(), vpux::DimsOrder::NHWC);

            EXPECT_EQ(ndOutputType.getMemSpace().getLeafName(), CMX_NAME);
            EXPECT_EQ(ndOutputType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

            const SmallVector<vpux::Bit> strides({262144_Bit, 16_Bit, 16384_Bit, 1024_Bit});
            const SmallVector<vpux::Bit> memStrides({262144_Bit, 16384_Bit, 1024_Bit, 16_Bit});
            EXPECT_EQ(ndOutputType.getStrides().raw(), strides);
            EXPECT_EQ(ndOutputType.getMemStrides().raw(), memStrides);

            EXPECT_EQ(ndOutputType.getElemTypeSize().count(), 16);
            EXPECT_EQ(ndOutputType.getTotalAllocSize().count(), 2*64*4*16);
            EXPECT_EQ(ndOutputType.getCompactAllocSize().count(), 2*64*4*16);

            EXPECT_EQ(ndOutputType.getShape(), vpux::ShapeRef({1, 64, 16, 16}));
            const SmallVector<int64_t> newShape({1, 32, 64, 8});
            const auto newFilterTypeShape = ndOutputType.changeShape(vpux::ShapeRef(newShape));
            EXPECT_EQ(newFilterTypeShape.getShape(), vpux::ShapeRef(newShape));

            EXPECT_TRUE(ndOutputType.getElementType().isa<mlir::Float16Type>());
            const auto newElemType = mlir::Float32Type::get(op.getContext());
            const auto newFilterTypeElemType = ndOutputType.changeElemType(newElemType);
            EXPECT_TRUE(newFilterTypeElemType.getElementType().isa<mlir::Float32Type>());

            EXPECT_EQ(ndOutputType.getDimsOrder(), DimsOrder::NHWC);
            const auto newDimsOrder = DimsOrder::NCHW;
            const auto newFilterTypeDimsOrder = ndOutputType.changeDimsOrder(newDimsOrder);
            EXPECT_EQ(newFilterTypeDimsOrder.getDimsOrder(), newDimsOrder);

            EXPECT_EQ(ndOutputType.getMemSpace().getLeafName(), CMX_NAME);
            const auto newMemSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
            const auto newFilterTypeMemSpace = ndOutputType.changeMemSpace(newMemSpace);
            EXPECT_EQ(newFilterTypeMemSpace.getMemSpace().getLeafName(), DDR_NAME);

            const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
            const SmallVector<int64_t> tileShape({1, 32, 32, 8});
            ASSERT_ANY_THROW(ndOutputType.extractDenseTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape)));
            const SmallVector<int64_t> pads({0, 0, 2, 2});
            ASSERT_ANY_THROW(ndOutputType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
        }
    }
}
