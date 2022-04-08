//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PerAxisTileOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::PerAxisTileOpAdaptor perAxisTile(operands, attrs);
    if (mlir::failed(perAxisTile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = perAxisTile.input().getType().cast<vpux::NDTypeInterface>();

    const auto axis = checked_cast<unsigned int>(perAxisTile.axis());
    const auto tiles = checked_cast<unsigned int>(perAxisTile.tiles());

    auto outShape = to_small_vector(inType.getShape().raw());

    if (axis > outShape.size()) {
        return errorAt(loc, "Axis is out of range. Available range [0, {0}), but got axis = {1}", outShape.size(),
                       axis);
    }

    outShape[axis] *= tiles;

    const auto outType = inType.changeShape(Shape(outShape));

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::PerAxisTileOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::TileParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axis()));
    builder.add_tiles(checked_cast<uint32_t>(tiles()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_TileParams});
}
