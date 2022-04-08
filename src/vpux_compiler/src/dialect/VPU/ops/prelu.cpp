//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(PReluOp op) {
    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto slopeType = op.negative_slope().getType().cast<vpux::NDTypeInterface>();
    const auto slopeShape = slopeType.getShape().raw();

    if (slopeShape.size() != 4) {
        return errorAt(op, "Tiling restrictions require slope to have a 4D shape, got size of {0}", slopeShape.size());
    }

    if (inShape[Dims4D::Act::C.ind()] != slopeShape[Dim(3).ind()]) {
        return errorAt(op,
                       "4D slope shape should have the last dim equal to the channel input dim, as broadcast with "
                       "numpy values is not supported: {0} != {1}",
                       inShape[Dims4D::Act::C.ind()], slopeShape[Dim(3).ind()]);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::PReluOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::PReluOpAdaptor prelu(operands, attrs);
    if (mlir::failed(prelu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = prelu.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::PReluOp::serialize(EMU::BlobWriter& writer) {
    const auto prelu = MVCNN::CreatePReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PReluParams);
    builder.add_nested_params(prelu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::PReluOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    TileInfo inputTile(getShape(input()));
    TileInfo slopeTile(getShape(negative_slope()));
    inputTile = outputTile;
    if (outputTile.shape[Dims4D::Act::C] != slopeTile.shape[Dims4D::Act::W]) {
        VPUX_THROW("Tiling per channel output is not supported for now, proposed {0} channel shape does not match the "
                   "slope value {1}.",
                   outputTile.shape[Dims4D::Act::C], slopeTile.shape[Dims4D::Act::W]);
    }

    return TilingInfo{{inputTile, slopeTile}};
}

void vpux::VPU::PReluOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing here
}

OutputTiling vpux::VPU::PReluOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
