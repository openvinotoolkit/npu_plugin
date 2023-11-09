//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;

void testExplicitDistributedAttr(llvm::StringLiteral inputIR, vpux::VPU::DistributedTensorAttr expectedDistributedAttr,
                                 vpux::ShapeRef shape, mlir::MLIRContext* ctx) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    for (auto& op : func.getOps()) {
        if (auto clusterOp = mlir::dyn_cast<vpux::VPU::ClusteredOpInterface>(op)) {
            auto distributedAttr = clusterOp.getExplicitDistributedTensorAttr(
                    shape, expectedDistributedAttr.getMode().getValue(), expectedDistributedAttr.getNumTiles(),
                    expectedDistributedAttr.getNumClusters(), expectedDistributedAttr.getAlignment(),
                    expectedDistributedAttr.getKernel(), expectedDistributedAttr.getPads(),
                    expectedDistributedAttr.getStrides(), expectedDistributedAttr.getUniformDistributedSegments());

            ASSERT_EQ(distributedAttr.getMemoryShapes(), expectedDistributedAttr.getMemoryShapes());
            ASSERT_EQ(distributedAttr.getMemoryOffsets(), expectedDistributedAttr.getMemoryOffsets());
            ASSERT_EQ(distributedAttr.getComputeShapes(), expectedDistributedAttr.getComputeShapes());
            ASSERT_EQ(distributedAttr.getComputeOffsets(), expectedDistributedAttr.getComputeOffsets());
        }
    }
}

using MLIR_GetExplicitDistributedTensorAttrTest = MLIR_UnitBase;

TEST_F(MLIR_GetExplicitDistributedTensorAttrTest, SWOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
                %0 = VPU.Interpolate(%arg0) {
                    attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>,
                    cube_coeff = -7.500000e-01 : f64,
                    mode = <LINEAR_ONNX>,
                    nearest_mode = <ROUND_PREFER_FLOOR>,
                    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                    shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
                    initial_input_dims_attr = [1, 1, 96, 160],
                    initial_output_dims_attr = [1, 1, 192, 320],
                    operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
                    scales_attr = [2.000000e+00, 2.000000e+00],
                    sizes_attr = [192, 320],
                    tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
                        : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
                return %0 : tensor<1x1x192x320xf16>
            }
        }
    )";
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    vpux::Shape outputShape = {1, 1, 192, 320};
    vpux::Shape inputShape = {1, 1, 96, 160};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 1, 49, 160});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 47, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto overlappedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, overlappedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, inputShape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 1, 96, 320});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 96, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, outputShape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 1, 192, 320});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, outputShape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributedTensorAttrTest, HWOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
                %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
                %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
                        activation_window_channel_length = 4 : i64,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        strides = [1, 1],
                        kernel_size = [1, 1]
                    } -> tensor<1x32x112x112xf16, {order = #NHWC}>
                return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
            }
        }
    )";
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    vpux::Shape shape = {1, 32, 112, 112};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1));
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 32, 56, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 56, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 32, 57, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 55, 0}});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);

        const auto overlappedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, overlappedMode, numTiles, kernel, pads, strides, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, shape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 32, 56, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 56, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 32, 112, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 32, 56, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 56, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 32, 112, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(numClusters.getInt(),
                                                                         SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, segDupMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributedTensorAttrTest, PermuteQuantOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

        !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
        module @test attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
            func.func @main(%arg0: tensor<1x224x3x224xf16, {order = #NHWC}>) -> tensor<1x224x4x224x!qElemType, {order = #NWCH}> {
                %0 = VPU.NCE.PermuteQuantize(%arg0) {
                    dstElemType = !qElemType,
                    dstOrder = #NWCH,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
                    ppe = #VPU.PPETask<
                        clamp_high = 255 : i64,
                        clamp_low = 0 : i64,
                        fp_prelu_alpha = 1.000000e+00 : f64,
                        lrelu_mult = 1 : i64,
                        lrelu_shift = 0 : i64,
                        mode = <NOOP>
                    >
                } -> tensor<1x224x4x224x!qElemType, {order = #NWCH}>
                return %0 : tensor<1x224x4x224x!qElemType, {order = #NWCH}>
            }
        }
    )";
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    vpux::Shape shape = {1, 224, 4, 224};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 1, 2}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1));
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 224, 4, 113});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 111}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto overlappedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, overlappedMode, numTiles, kernel, pads, strides, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, shape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 224, 4, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 112}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 224, 4, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 224, 4, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 112}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 224, 4, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(numClusters.getInt(),
                                                                         SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, segDupMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributedTensorAttrTest, ConcatOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x48x32x32xf16, {order = #NHWC}>,
                            %arg1: tensor<1x48x32x32xf16, {order = #NHWC}>)
                    -> tensor<1x96x32x32xf16, {order = #NHWC}> {
                %0 = VPU.Concat(%arg0, %arg1) {
                    static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]
                } : tensor<1x48x32x32xf16, {order = #NHWC}>,
                    tensor<1x48x32x32xf16, {order = #NHWC}>
                        -> tensor<1x96x32x32xf16, {order = #NHWC}>

                return %0 : tensor<1x96x32x32xf16, {order = #NHWC}>
            }
        }
    )";
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    vpux::Shape shape = {1, 96, 32, 32};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1));
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 96, 17, 32});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto overlappedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, overlappedMode, numTiles, kernel, pads, strides, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, shape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 96, 16, 32});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 16, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 96, 32, 32});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}
