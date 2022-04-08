//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RollOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::RollOpAdaptor roll(operands, attrs);
    if (mlir::failed(roll.verify(loc))) {
        return mlir::failure();
    }

    const auto shiftContent = roll.shift().getDefiningOp<Const::DeclareOp>().content();
    const auto inShapeShift = getShape(roll.shift());

    if (!shiftContent.isSplat() && inShapeShift.size() == 1) {
        auto shiftData = IE::constInputToData(loc, roll.shift());
        if (mlir::failed(shiftData)) {
            return mlir::failure();
        }

        auto axesData = IE::constInputToData(loc, roll.axes());
        if (mlir::failed(axesData)) {
            return mlir::failure();
        }

        auto shiftShape = shiftData.getValue();
        auto axesShape = axesData.getValue();

        if (shiftShape.size() != axesShape.size()) {
            return errorAt(loc,
                           "If shift is a 1D vector, axes must be a 1D tensor of the same size. Got shift size {0} and "
                           "axes size {1}.",
                           shiftShape.size(), axesShape.size());
        }
    }

    const auto inType = roll.data().getType().cast<vpux::NDTypeInterface>();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::RollOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::RollParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RollParams});
}
