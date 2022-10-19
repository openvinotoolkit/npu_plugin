//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MemPermuteOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::MemPermuteOpAdaptor mem_permute(operands, attrs);
    if (mlir::failed(mem_permute.verify(loc))) {
        return mlir::failure();
    }

    VPU::inferPermuteReturnTypes(mem_permute.input(), mem_permute.mem_perm().getValue(),
                                 mem_permute.dst_order().getValue(), inferredReturnTypes);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::MemPermuteOp::serialize(EMU::BlobWriter& writer) {
    const mlir::AffineMap inputOrderMap = DimsOrder::fromValue(input()).toAffineMap(this->getContext());
    const mlir::AffineMap permMem = mem_perm();
    const mlir::AffineMap outputOrderMapInv =
            inversePermutation(DimsOrder::fromValue(output()).toAffineMap(this->getContext()));

    const mlir::AffineMap permLog = outputOrderMapInv.compose(permMem.compose(inputOrderMap));

    const auto permLogOrder = DimsOrder::fromAffineMap(permLog);
    const auto orderUPA = writer.createVector(irange(permLogOrder.numDims()) | transformed([&](int64_t idx) {
                                                  return checked_cast<int32_t>(permLogOrder.dimAt(idx).ind());
                                              }));

    MVCNN::PermuteNDParamsBuilder builder(writer);
    builder.add_permute_nd_order(orderUPA);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PermuteNDParams});
}

InputTiling vpux::VPU::MemPermuteOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
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
    return TilingInfo{curTile};
}

void vpux::VPU::MemPermuteOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}
