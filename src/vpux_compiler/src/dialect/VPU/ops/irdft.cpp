//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

struct IRDFTInputData final {
    SmallVector<int64_t> axes;
    SmallVector<int64_t> signal_size;
};

mlir::FailureOr<IRDFTInputData> extractData(VPU::IRDFTOpAdaptor op) {
    if (op.axes_attr() && op.signal_size_attr()) {
        auto axes = parseIntArrayAttr<int64_t>(op.axes_attr());
        auto signal_size = parseIntArrayAttr<int64_t>(op.signal_size_attr());
        return IRDFTInputData{std::move(axes), std::move(signal_size)};
    }
    return mlir::failure();
}

}  // namespace

mlir::LogicalResult vpux::VPU::IRDFTOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,

                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));
    VPU::IRDFTOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }
    const auto inputData = extractData(op);
    SmallVector<int64_t> axes = inputData.getValue().axes;
    SmallVector<int64_t> signal_size = inputData.getValue().signal_size;

    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    auto outShape = to_small_vector(inType.getShape());
    // delete last size, 2 in this case
    outShape.pop_back();

    const auto last_axis = axes.back();
    outShape[last_axis] = (outShape[last_axis] - 1) * 2;

    for (size_t i = 0; i < axes.size(); ++i) {
        const int64_t current_axis = axes[i];
        if (signal_size[i] != -1) {
            outShape[current_axis] = signal_size[i];
        }
    }

    auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

void vpux::VPU::IRDFTOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::IRDFTOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("IRDFT is not implemented in UPA Tasks.");
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::IRDFTOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto curTile = outputTile;
    SmallVector<int64_t> axes = parseIntArrayAttr<int64_t>(axes_attr());
    const auto inShape = getShape(input());
    for (auto axis : axes) {
        curTile.shape[Dim(axis)] = inShape[Dim(axis)];
    }
    // input have 1 extra dimension complex number representation
    curTile.shape.push_back(2);
    curTile.offsets.push_back(0);
    curTile.axis.push_back(1);
    return TilingInfo{curTile};
}

void vpux::VPU::IRDFTOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

// Based on getSWLayerTilingStrategy logic, need to avoid tiling on all axes.

OutputTiling vpux::VPU::IRDFTOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    // eliminate axes from possible tiling dims
    SmallVector<int64_t> axes = parseIntArrayAttr<int64_t>(axes_attr());
    return getSWLayerTilingStrategyWithAxesExclusion(getOperation(), tilingMode, log, axes);
}
