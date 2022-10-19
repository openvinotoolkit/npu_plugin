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

TEST(MLIR_NDTypeInterface, SparseTensorType_Weights) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const Shape shape{64, 16, 3, 3};
    const auto data = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMap = mlir::RankedTensorType::get(shape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto sparseTensorType = VPU::SparseTensorType::get(data, sparsityMap);

    const auto ndType = sparseTensorType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 3 * 3);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({2304_Bit, 144_Bit, 48_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 19584);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 19584);

    const SmallVector<int64_t> newShape({32, 16, 3, 3});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto sparseChangedShape = changedShape.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(sparseChangedShape.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());
    const auto sparseChangedElemType = changedElemType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(sparseChangedElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(sparseChangedElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());

    const SmallVector<int64_t> newShape2({32, 16, 5, 5});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::Float64Type>());
    const auto sparseChangedShapeElemType = changedShapeElemType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float64Type>());
    EXPECT_TRUE(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = changedDimsOrder.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(sparseChangedDimsOrder.getData().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getSparsityMap().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = changedMemSpace.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = changedMemSpace.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);

    const SmallVector<Bit> newStrides({4608_Bit, 144_Bit, 48_Bit, 16_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 16, 3, 3});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = tiledType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    const SmallVector<int64_t> paddedShape({64, 16, 5, 5});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = paddedType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(sparsePaddedType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
}

TEST(MLIR_NDTypeInterface, SparseTensorType_Activation) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape seTableShape{1, 1, 32, 32};
    const auto data = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto sparsityMap = mlir::RankedTensorType::get(shape.raw(), mlir::IntegerType::get(&ctx, 1));
    const auto storageElementTable = mlir::RankedTensorType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32));
    const auto sparseTensorType = VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);

    const auto ndType = sparseTensorType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 1 * 64 * 32 * 32);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), DimsOrder::NCHW);

    EXPECT_EQ(ndType.getMemSpace(), nullptr);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 143360);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 143360);

    const SmallVector<int64_t> newShape({1, 64, 32, 16});
    const SmallVector<int64_t> newSETableShape({1, 1, 32, 16});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto sparseChangedShape = changedShape.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(sparseChangedShape.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(newSETableShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());
    const auto sparseChangedElemType = changedElemType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(sparseChangedElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(sparseChangedElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedElemType.getStorageElementTable().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());

    const SmallVector<int64_t> newShape2({1, 32, 32, 32});
    const SmallVector<int64_t> newSETableShape2({1, 1, 32, 32});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::Float64Type>());
    const auto sparseChangedShapeElemType = changedShapeElemType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(newSETableShape2));
    EXPECT_TRUE(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float64Type>());
    EXPECT_TRUE(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedShapeElemType.getStorageElementTable().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = changedDimsOrder.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(sparseChangedDimsOrder.getData().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getSparsityMap().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getStorageElementTable().cast<NDTypeInterface>().getDimsOrder(), DimsOrder::NCHW);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = changedMemSpace.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = changedMemSpace.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);

    const SmallVector<Bit> newStrides({2097152_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({1, 64, 16, 32});
    const SmallVector<int64_t> tileSETableShape({1, 1, 16, 32});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = tiledType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 2, 1, 1});
    const SmallVector<int64_t> padAfter({0, 2, 1, 1});
    const SmallVector<int64_t> paddedShape({1, 68, 34, 34});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 34, 34});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = paddedType.dyn_cast<VPU::SparseTensorType>();
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(sparsePaddedType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(paddedSETableShape));
}
