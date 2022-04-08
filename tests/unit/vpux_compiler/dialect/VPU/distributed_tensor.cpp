//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

TEST(MLIR_NDTypeInterface, SegmentedDistributedTensorType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionMode, numTiles, nullptr, nullptr, nullptr,
                                                                 numClusters, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 13 * 16);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({212992_Bit, 16384_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 4 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 4 * 16);

    const SmallVector<int64_t> newShape({1, 32, 52, 8});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());

    const SmallVector<int64_t> newShape2({1, 32, 26, 16});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::IntegerType>());

    const auto newDimsOrder = DimsOrder::NCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    EXPECT_ANY_THROW(ndType.extractDenseTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape)));
    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape), vpux::ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

// Test out combination of SEGMENTED | DUPLICATED mode

TEST(MLIR_NDTypeInterface, SegmentedDuplicatedDistributedTensorType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionMode, numTiles, nullptr, nullptr, nullptr,
                                                                 numClusters, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 13 * 16);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({212992_Bit, 16384_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 13 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 13 * 16);

    const SmallVector<int64_t> newShape({1, 32, 52, 8});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());
    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());

    const SmallVector<int64_t> newShape2({1, 32, 32, 32});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::IntegerType>());

    const auto newDimsOrder = DimsOrder::NCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    EXPECT_ANY_THROW(ndType.extractDenseTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape)));
    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape), vpux::ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST(MLIR_ClusterShapeUtils, SegmentedDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterStridedShapes();
    for (const auto p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST(MLIR_ClusterShapeUtils, SegmentedDuplicatedDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const Strides expectedStrides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto perClusterStridedShapes = distributedType.getPerClusterStridedShapes();
    for (const auto p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
}

TEST(MLIR_ClusterShapeUtils, OverlappedDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                                getIntAttr(&ctx, 0), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedType.getPerClusterStridedShapes();
        for (const auto p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[0]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }
    }

    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 5, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 2, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 6, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedType.getPerClusterStridedShapes();
        for (const auto p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }
    }

    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 2, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 5, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedType.getPerClusterStridedShapes();
        for (const auto p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }
    }
}

TEST(MLIR_ClusterShapeUtils, DISABLED_AlignedTensorDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // Single axis H alignment, H SEGMENTED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 9, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 60, 18, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 18 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 18 * 16);
    }

    // Multiple axis H and K alignment, H SEGMENTED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 9, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 18, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 18 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 18 * 16);
    }

    // Single axis H alignment, DUPLICATED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 60, 63, 16}), Shape({1, 60, 63, 16}), Shape({1, 60, 63, 16}), Shape({1, 60, 63, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 0, 0}), Shape({0, 0, 0, 0}), Shape({0, 0, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 60, 63, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
    }

    // Single axis H alignment, SEGMENTED|DUPLICATED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 9, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 60, 18, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
    }

    // Multiple axis H and K alignment, SEGMENTED|DUPLICATED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 9, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 18, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 63 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 63 * 16);
    }

    // Single axis K alignment, SEGMENTED mode, K tiling
    {
        const auto shape = SmallVector<int64_t>({1, 110, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 16, 59, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0}), Shape({0, 96, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 32, 59, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
    }

    // Single axis K alignment, SEGMENTED mode, K tiling, invalid 4 cluster tiling
    {
        const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        EXPECT_ANY_THROW(distributedType.getPerClusterComputeShapes());
        EXPECT_ANY_THROW(distributedType.getPerClusterComputeShapeOffsets());
        EXPECT_ANY_THROW(distributedType.getLargestCompactShape());
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_ANY_THROW(distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_ANY_THROW(ndType.getTotalAllocSize().count());
        EXPECT_ANY_THROW(ndType.getCompactAllocSize().count());
    }

    // Single axis K alignment, SEGMENTED mode, K tiling, valid 3 cluster tiling
    {
        const auto numClustersAttr = getIntAttr(&ctx, 3);
        const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets({Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 32, 59, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
    }

    // Single axis K alignment, SEGMENTED|DUPLICATED mode, K tiling, valid 3 cluster tiling
    {
        const auto numClustersAttr = getIntAttr(&ctx, 3);
        const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets({Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 32, 59, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 96 * 59 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 96 * 59 * 16);
    }

    // Single axis K alignment, OVERLAPPED mode, H tiling
    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
        const auto shape = SmallVector<int64_t>({1, 60, 13, 15});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 15}), Shape({1, 64, 5, 15}), Shape({1, 64, 5, 15}), Shape({1, 64, 2, 15})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 5, 15}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 5 * 15);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 5 * 15);
    }

    // Single axis W alignment, OVERLAPPED mode, H tiling
    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
        const auto shape = SmallVector<int64_t>({1, 60, 13, 15});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 1, 16}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 60, 4, 16}), Shape({1, 60, 5, 16}), Shape({1, 60, 5, 16}), Shape({1, 60, 2, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 60, 5, 16}));
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 5 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 5 * 16);
    }
}

TEST(MLIR_ClusterShapeUtilsDeathTest, AlignedTensorDistribution) {
    testing::GTEST_FLAG(death_test_style) = "threadsafe";
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // Single axis H alignment, OVERLAPPED mode, H tiling, invalid alignment axis same as tiling axis
    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
        const auto shape = SmallVector<int64_t>({1, 60, 59, 15});
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, alignment, &ctx);

#if !defined(NDEBUG)
        EXPECT_DEATH(VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr),
                     "Overlapped cluster tiling does not support alignment on the same axis used for tiling");
#else
        const auto distributedType =
                VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);
        EXPECT_TRUE(VPU::verify(mlir::detail::getDefaultDiagnosticEmitFn(&ctx), distributedType.getDistribution(),
                                distributedType.getShape().raw())
                            .failed());
#endif
    }
}
