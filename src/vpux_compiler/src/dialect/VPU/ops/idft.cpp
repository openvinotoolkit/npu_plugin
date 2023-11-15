//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::IDFTOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::IDFTOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(op.axes_attr());
    auto signalSize = parseIntArrayAttr<int64_t>(op.signal_size_attr());

    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    auto outShape = to_small_vector(inType.getShape());

    for (size_t i = 0; i < axes.size(); ++i) {
        if (signalSize[i] != -1) {
            outShape[axes[i]] = signalSize[i];
        }
    }

    auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::IDFTOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("IDFT is not implemented in UPA Tasks.");
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::IDFTOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto curTile = outputTile;
    SmallVector<int64_t> axes = parseIntArrayAttr<int64_t>(axes_attr());
    const auto inShape = getShape(input());
    for (auto axis : axes) {
        curTile.shape[Dim(axis)] = inShape[Dim(axis)];
    }
    TileInfo twiddleTile(getShape(twiddle_factors()));
    return TilingInfo{{std::move(curTile), std::move(twiddleTile)}};
}

void vpux::VPU::IDFTOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::IDFTOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = getOperation();
    // eliminate axes from possible tiling dims
    auto axes = parseIntArrayAttr<int64_t>(axes_attr());
    // add last axis to not allowed split as represent the complex number
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    axes.push_back(outputShape.size() - 1);

    return getSWLayerTilingStrategy(op, tilingMode, log, getMaxNumTilesWithAxesExclusion(op, axes));
}
