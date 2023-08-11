//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PermuteQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PermuteQuantizeOpAdaptor permute_quantize(operands, attrs);
    if (mlir::failed(permute_quantize.verify(loc))) {
        return mlir::failure();
    }

    mlir::Value input = permute_quantize.input();
    mlir::AffineMap memPerm = permute_quantize.mem_perm();
    mlir::AffineMap dstOrder = permute_quantize.dst_order();
    const auto padBegin = parseIntArrayAttr<int64_t>(permute_quantize.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(permute_quantize.pads_end());

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dstOrder);
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();

    const auto newExpandedInType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    const auto inShapeExpanded = newExpandedInType.getShape();

    const auto inMemShape = inOrder.toMemoryOrder(inShapeExpanded);
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);
    const auto outType = inType.changeDimsOrder(outOrder).changeShape(outShape);

    const auto dstElemType = permute_quantize.dstElemType();

    const auto outTypeFin = outType.changeElemType(dstElemType);

    inferredReturnTypes.push_back(outTypeFin);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::PermuteQuantizeOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("PermuteQuantizeOp implemented just on 37xx.");
}

InputTiling vpux::VPU::PermuteQuantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    mlir::AffineMap memPerm = mem_perm();
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    const auto inShape = getShape(input());
    const auto inOrder = DimsOrder::fromValue(input());
    const auto outOrder = DimsOrder::fromValue(output());
    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        // take in consideration input and output shape vector order not map with memory order
        auto idxOrdIn = inOrder.dimAt(perm.dimAt(ind).ind());
        auto idxOrdOut = outOrder.dimAt(ind);
        curTile.shape[idxOrdIn] = outputTile.shape[idxOrdOut];
        curTile.offsets[idxOrdIn] = outputTile.offsets[idxOrdOut];
        curTile.axis[idxOrdIn] = outputTile.axis[idxOrdOut];
    }
    const auto iType = input().getType().cast<vpux::NDTypeInterface>();
    const auto oType = output().getType().cast<vpux::NDTypeInterface>();

    curTile.shape[Dims4D::Act::C] = iType.getShape()[Dims4D::Act::C];
    if (outputTile.shape[Dims4D::Act::C] != oType.getShape()[Dims4D::Act::C]) {
        VPUX_THROW("Unsupported Tile For PermuteQuantizeExpandOver expanded Channel  outTile: '{0}' InTile: {1}",
                   outputTile, curTile);
    }

    return TilingInfo{curTile};
}

void vpux::VPU::PermuteQuantizeOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

OutputTiling vpux::VPU::PermuteQuantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
