//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FullyConnectedOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FullyConnectedOpAdaptor fullyConnected(operands, attrs);
    if (mlir::failed(fullyConnected.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = fullyConnected.input().getType().cast<vpux::NDTypeInterface>();
    const auto weightsType = fullyConnected.weights().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto weightsShape = weightsType.getShape().raw();
    const auto inRank = inShape.size();
    const auto weightsRank = weightsShape.size();

    if (weightsRank != 2 || inRank != 2) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    outShape.push_back(inShape[0]);
    outShape.push_back(weightsShape[0]);

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void vpux::VPU::FullyConnectedOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::FullyConnectedOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::FullyConnectedParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FullyConnectedParams});
}
