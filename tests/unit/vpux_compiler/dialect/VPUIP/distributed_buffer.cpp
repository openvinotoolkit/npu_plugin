//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

TEST(MLIR_NDTypeInterface, SegmentedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 13);

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
    const auto chnagedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(chnagedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElementType.getElementType().isa<mlir::Float32Type>());

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(changedShapeAndElementType.getElementType().isa<mlir::Float32Type>());

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);
    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> newStrides2({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({638976_Bit, 16_Bit, 32768_Bit, 1024_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST(MLIR_NDTypeInterface, SegmentedDuplicatedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 13);

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
    const auto chnagedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(chnagedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElementType.getElementType().isa<mlir::Float32Type>());

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(changedShapeAndElementType.getElementType().isa<mlir::Float32Type>());

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);

    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> newStrides2({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({638976_Bit, 16_Bit, 32768_Bit, 1024_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST(MLIR_NDTypeInterface, CompressedSegmentedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({64, 16, 1, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::OYXI.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({16, 1, 16, 16});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const SmallVector<int64_t> numElems(64, 15);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, makeArrayRef(numElems));

    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto compressionScheme = VPUIP::CompressionSchemeAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                     numElemsAttr, getIntAttr(&ctx, alignment));

    const auto ndType = VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr,
                                                          compressionScheme)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));
    EXPECT_EQ(ndType.getTotalAllocSize().count(),
              (64 / 4) * (15 * sizeof(float16) + 2));  // weight-set size aligned to 16 bytes
    EXPECT_EQ(ndType.getCompactAllocSize().count(),
              (64 / 4) * (15 * sizeof(float16) + 2));  // weight-set size aligned to 16 bytes

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 16, 1, 1});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto distTiledType = tiledType.dyn_cast<VPUIP::DistributedBufferType>();
    ASSERT_TRUE(distTiledType != nullptr);
    auto tiledNumElems = distTiledType.getCompressionScheme().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);
}

TEST(MLIR_NDTypeInterface, CompressedDuplicatedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({64, 80, 1, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::OYXI.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({80, 1, 80, 80});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, makeArrayRef(numElems));
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto compressionScheme = VPUIP::CompressionSchemeAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                     numElemsAttr, getIntAttr(&ctx, alignment));

    const auto ndType = VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr,
                                                          compressionScheme)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));

    int64_t totalByteSize = 0;
    for (auto elems : numElems) {
        int64_t weightSetByteSize = elems * sizeof(float16);
        totalByteSize += vpux::alignVal(weightSetByteSize, alignment);
    }
    EXPECT_EQ(ndType.getTotalAllocSize().count(), totalByteSize);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), totalByteSize);

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 80, 1, 1});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto distTiledType = tiledType.dyn_cast<VPUIP::DistributedBufferType>();
    ASSERT_TRUE(distTiledType != nullptr);
    auto tiledNumElems = distTiledType.getCompressionScheme().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);
}

TEST(MLIR_ClusterShapeUtils, SegmentedBufferDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    {
        // SOH
        const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
        const auto elemType = mlir::Float16Type::get(&ctx);

        const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[0]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
    }

    {
        // SOH, W strided
        const auto shape = SmallVector<int64_t>({1, 64, 16, 4});
        const auto elemType = mlir::Float16Type::get(&ctx);

        const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
        const auto elemStrides = SmallVector<int64_t>({64 * 4 * 2 * 16, 1, 64 * 4 * 2, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 4}), Shape({1, 64, 4, 4}), Shape({1, 64, 4, 4}), Shape({1, 64, 4, 4})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 4}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const Shape expectedShape({1, 64, 4, 4});
        const Strides expectedStrides({32768_Bit, 16_Bit, 8192_Bit, 1024_Bit});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& stridedShape : perClusterStridedShapes) {
            EXPECT_EQ(stridedShape.shape, expectedShape);
            EXPECT_EQ(stridedShape.strides, expectedStrides);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShape);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShape);
            EXPECT_EQ(stridedShape.strides, expectedStrides);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 8 * 4 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 4 * 2);
    }
}

TEST(MLIR_ClusterShapeUtils, SegmentedDuplicatedBufferDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    // SOH duplicated
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr,
                                                                 nullptr, numClustersAttr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const Strides expectedStrides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 13 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
}

TEST(MLIR_ClusterShapeUtils, OverlappedBufferDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    {
        // SOH overlapped, 1x1s1p0
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                                getIntAttr(&ctx, 0), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[0]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
    }

    {
        // SOH overlapped, 3x3s1p1
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 5, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 2, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 6, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 6 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 6 * 16 * 2);
    }

    {
        // SOH overlapped, 3x3s2p1
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 2, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 5, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 5 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 5 * 16 * 2);
    }
}

TEST(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // Single axis H alignment, H SEGMENTED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 18 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 18 * 16);
    }

    // Multiple axis H and K alignment, H SEGMENTED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 18 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 18 * 16);
    }

    // Single axis H alignment, DUPLICATED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
    }

    // Single axis H alignment, SEGMENTED|DUPLICATED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
    }

    // Multiple axis H and K alignment, SEGMENTED|DUPLICATED mode
    {
        const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 63 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 63 * 16);
    }

    // Single axis K alignment, SEGMENTED mode, K tiling
    {
        const auto shape = SmallVector<int64_t>({1, 110, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({110 * 16 * 59, 1, 110 * 16, 110});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
    }

    // Single axis K alignment, SEGMENTED mode, K tiling, invalid 4 cluster tiling
    {
        const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({96 * 16 * 59, 1, 96 * 16, 96});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        EXPECT_ANY_THROW(distributedType.getPerClusterComputeShapes());
        EXPECT_ANY_THROW(distributedType.getPerClusterComputeShapeOffsets());
        EXPECT_ANY_THROW(distributedType.getLargestCompactShape());
        const auto numClusters = distributedType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_ANY_THROW(distributedType.getCompactShape(clusterIdx));
        }

        const auto ndType = distributedType.dyn_cast<vpux::NDTypeInterface>();
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_ANY_THROW(ndType.getTotalAllocSize().count());
        EXPECT_ANY_THROW(ndType.getCompactAllocSize().count());
    }

    // Single axis K alignment, SEGMENTED mode, K tiling, valid 3 cluster tiling
    {
        const auto numClustersAttr = getIntAttr(&ctx, 3);
        const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({96 * 16 * 59, 1, 96 * 16, 96});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
    }

    // Single axis K alignment, SEGMENTED|DUPLICATED mode, K tiling, valid 3 cluster tiling
    {
        const auto numClustersAttr = getIntAttr(&ctx, 3);
        const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
        const auto elemStrides = SmallVector<int64_t>({96 * 16 * 59, 1, 96 * 16, 96});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(
                distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

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
        const auto elemStrides = SmallVector<int64_t>({60 * 15 * 13, 1, 60 * 15, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

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
        const auto elemStrides = SmallVector<int64_t>({60 * 15 * 13, 1, 60 * 15, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 1, 16}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, alignment, &ctx);
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
        ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

        EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 5 * 16);
        EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 5 * 16);
    }
}

TEST(MLIR_ClusterShapeUtilsDeathTest, AlignedBufferDistribution) {
    testing::GTEST_FLAG(death_test_style) = "threadsafe";
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

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
        const auto elemStrides = SmallVector<int64_t>({60 * 15 * 59, 1, 60 * 15, 60});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(dimsOrder, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, alignment, &ctx);

#if !defined(NDEBUG)
        EXPECT_DEATH(VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr),
                     "Overlapped cluster tiling does not support alignment on the same axis used for tiling");
#else
        const auto distributedType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);
        EXPECT_TRUE(VPU::verify(mlir::detail::getDefaultDiagnosticEmitFn(&ctx), distributedType.getDistribution(),
                                distributedType.getShape().raw())
                            .failed());
#endif
    }
}

TEST(MLIR_ClusterShapeUtils, StridedBufferDistribution) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // SOH, K striding
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13, 1, 64 * 2 * 16, 64 * 2});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {32768_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * 4 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
    }

    // Overlapped, K striding
    {
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1), &ctx);
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));

        const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13, 1, 64 * 2 * 16, 64 * 2});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads,
                                                                     strides, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 2, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 5, 16}));

        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {163840_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {163840_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {65536_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * 5 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 5 * 16 * 2);
    }

    // SOK, H striding
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 16, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{425984_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {425984_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {425984_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                    {425984_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * (13 * 2) * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
    }

    // SOK, no duplication, H striding
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 16, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{106496_Bit, 16_Bit, 4096_Bit, 256_Bit},
                                                    {106496_Bit, 16_Bit, 4096_Bit, 256_Bit},
                                                    {106496_Bit, 16_Bit, 4096_Bit, 256_Bit},
                                                    {106496_Bit, 16_Bit, 4096_Bit, 256_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 16 * (13 * 2) * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 16 * 13 * 16 * 2);
    }

    // SOK, weights set (IC*Kw*Kh) striding
    {
        // Original size is {64, 3, 3, 3} but all values are packed and
        // aligned to 16 bytes so their memory access is aligned to 16 bytes.
        const auto shape = SmallVector<int64_t>({64, 1, 1, 27});
        const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::OYXI.toAffineMap(&ctx));

        const auto elemStrides = SmallVector<int64_t>({32, 1, 32, 32});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({16, 1, 1, 27}), Shape({16, 1, 1, 27}), Shape({16, 1, 1, 27}), Shape({16, 1, 1, 27})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({16, 0, 0, 0}), Shape({32, 0, 0, 0}), Shape({48, 0, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({16, 1, 1, 27}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{512_Bit, 16_Bit, 512_Bit, 512_Bit},
                                                    {512_Bit, 16_Bit, 512_Bit, 512_Bit},
                                                    {512_Bit, 16_Bit, 512_Bit, 512_Bit},
                                                    {512_Bit, 16_Bit, 512_Bit, 512_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 16 * 32 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 16 * 27 * 2);
    }

    // SOK, K striding
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 2 * 16, 64 * 2});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{425984_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {425984_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {425984_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {425984_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * 13 * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
    }

    // SOK, H + K striding
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13 * 2, 1, 64 * 2 * 16, 64 * 2});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{851968_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {851968_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {851968_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                    {851968_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * (13 * 2) * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
    }

    // SOH, W + K striding
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 2 * 13, 1, 64 * 2 * 16 * 2, 64 * 2});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        const SmallVector<Strides> expectedStrides({{262144_Bit, 16_Bit, 65536_Bit, 2048_Bit},
                                                    {262144_Bit, 16_Bit, 65536_Bit, 2048_Bit},
                                                    {262144_Bit, 16_Bit, 65536_Bit, 2048_Bit},
                                                    {65536_Bit, 16_Bit, 65536_Bit, 2048_Bit}});
        const auto perClusterStridedShapes = distributedBufferType.getPerClusterStridedShapes();
        for (const auto& p : perClusterStridedShapes | indexed) {
            const auto cluster = p.index();
            const auto stridedShape = p.value();
            EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
        }
        const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
        EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
        EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
            EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
            EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
        }

        EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * (4 * 2) * 16 * 2);
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
    }

    // SOH, H striding - unsupported
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 16, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        // const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
        //                                             {131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
        //                                             {131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
        //                                             { 32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
        EXPECT_ANY_THROW(distributedBufferType.getPerClusterStridedShapes());
        // 64 * 16 * (4 * 2) * 2
        EXPECT_ANY_THROW(distributedBufferType.getTotalAllocSize().count());
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 16 * 4 * 2);
    }

    // SOK, no duplication, K striding - unsupported
    {
        const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13, 1, 64 * 2 * 16, 64 * 2});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr, &ctx);

        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
        const auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr,
                                                                     nullptr, nullptr, numClustersAttr, nullptr, &ctx);
        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        const auto perClusterShapes = distributedBufferType.getPerClusterComputeShapes();
        const SmallVector<Shape> expectedShapes(
                {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
        for (const auto shapePair : zip(perClusterShapes, expectedShapes)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto perClusterTensorOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
        const SmallVector<Shape> expectedOffsets(
                {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
        for (const auto shapePair : zip(perClusterTensorOffsets, expectedOffsets)) {
            EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
        }
        const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
        EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
        const auto numClusters = distributedBufferType.getDistribution().num_clusters().getInt();
        for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
        }

        // const SmallVector<Strides> expectedStrides({{106496_Bit, 16_Bit, 8192_Bit, 512_Bit},
        //                                             {106496_Bit, 16_Bit, 8192_Bit, 512_Bit},
        //                                             {106496_Bit, 16_Bit, 8192_Bit, 512_Bit},
        //                                             {106496_Bit, 16_Bit, 8192_Bit, 512_Bit}});
        EXPECT_ANY_THROW(distributedBufferType.getPerClusterStridedShapes());
        // (16 * 2) * 16 * 13 * 2
        EXPECT_ANY_THROW(distributedBufferType.getTotalAllocSize().count());
        EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 16 * 13 * 16 * 2);
    }
}
