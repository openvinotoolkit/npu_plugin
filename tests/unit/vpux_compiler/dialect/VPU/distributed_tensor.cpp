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

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    EXPECT_ANY_THROW(ndType.extractDenseTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape)));
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

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    EXPECT_ANY_THROW(ndType.extractDenseTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape)));
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
    }
}
