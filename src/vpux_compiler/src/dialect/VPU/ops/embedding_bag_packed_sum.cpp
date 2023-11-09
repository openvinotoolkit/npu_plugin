//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EmbeddingBagPackedSumOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange, mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::EmbeddingBagPackedSumOpAdaptor embeddingBag(operands, attrs);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto embTableType = embeddingBag.emb_table().getType().cast<vpux::NDTypeInterface>();
    const auto embTableShape = embTableType.getShape().raw();
    const auto indicesType = embeddingBag.indices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape().raw();
    int64_t batchSize = indicesShape[0];

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < embTableShape.size(); i++) {
        outShape.emplace_back(embTableShape[i]);
    }
    outShape[0] = batchSize;
    const auto outType = embTableType.changeShape(Shape(outShape));

    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::EmbeddingBagPackedSumOp::serialize(vpux::EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("EmbeddingBagPackedSum does not support UPA task.");
}
