//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputSelectBoxesOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputSelectBoxesOpAdaptor selectBoxes(operands, attrs);
    if (mlir::failed(selectBoxes.verify(loc))) {
        return mlir::failure();
    }

    const auto indicesType = selectBoxes.indices().getType().cast<NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();

    const auto numPriors = indicesShape[Dim(2)];
    const auto topK = selectBoxes.top_k();

    const auto numOutBoxes = std::min(topK, numPriors);

    const auto boxesType = selectBoxes.decoded_boxes().getType().cast<NDTypeInterface>();
    const auto outputShape = SmallVector<int64_t>{indicesShape[Dim(0)], indicesShape[Dim(1)], numOutBoxes, 4};
    const auto outputType = boxesType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputSelectBoxesOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputSelectBoxesOp is not supported by EMU");
}

//
// TilingBuilderOpInterface
//

using DimType = Shape::ValueType;

TileInfo tileDecodedBoxes(mlir::Value decodedBoxes, DimType numClasses, DimType classesOffset) {
    const auto decodedBoxesShape = getShape(decodedBoxes);

    auto tile = TileInfo(decodedBoxesShape);
    const auto numLocClasses = decodedBoxesShape[Dims4D::Act::C];
    if (numLocClasses > 1) {
        tile.shape[Dims4D::Act::C] = numClasses;
        tile.offsets[Dims4D::Act::C] = classesOffset;
    }

    return tile;
}

TileInfo tileIndices(mlir::Value indices, DimType numClasses, DimType classesOffset) {
    const auto indicesShape = getShape(indices);

    auto tile = TileInfo(indicesShape);
    const auto Dim3D_H = Dim(1);  // CHW
    tile.shape[Dim3D_H] = numClasses;
    tile.offsets[Dim3D_H] = classesOffset;

    return tile;
}

TileInfo tileSizes(mlir::Value sizes, DimType numClasses, DimType classesOffset) {
    const auto shape = getShape(sizes);

    auto tile = TileInfo(shape);
    const auto Dim2D_C = Dim(1);  // NC
    tile.shape[Dim2D_C] = numClasses;
    tile.offsets[Dim2D_C] = classesOffset;

    return tile;
}

InputTiling vpux::VPU::DetectionOutputSelectBoxesOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                       vpux::Logger /*log*/) {
    // outputTile [Batch, numClasses, numPriors, 4]
    const auto numClasses = outputTile.shape[Dims4D::Act::C];
    const auto classesOffset = outputTile.offsets[Dims4D::Act::C];

    const auto decodedBoxesTile = tileDecodedBoxes(decoded_boxes(), numClasses, classesOffset);
    const auto indicesTile = tileIndices(indices(), numClasses, classesOffset);
    const auto sizesTile = tileSizes(sizes(), numClasses, classesOffset);

    return InputTiling{{decodedBoxesTile, indicesTile, sizesTile}};
}

void vpux::VPU::DetectionOutputSelectBoxesOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    return;
}

mlir::FailureOr<OutputTiling> vpux::VPU::DetectionOutputSelectBoxesOp::getTilingStrategy(TilingMode tilingMode,
                                                                                         Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
