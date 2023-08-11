//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PadOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PadOpAdaptor pad(operands, attrs);
    if (mlir::failed(pad.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = pad.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    auto padBegin = IE::extractPads(loc, pad.pads_begin(), pad.pads_begin_attr(), inputShape);
    if (mlir::failed(padBegin)) {
        return mlir::failure();
    }
    const auto padEnd = IE::extractPads(loc, pad.pads_end(), pad.pads_end_attr(), inputShape);
    if (mlir::failed(padEnd)) {
        return mlir::failure();
    }
    if (pad.mode() == IE::PadMode::CONSTANT && pad.pad_value() == nullptr && !pad.pad_value_attr().hasValue()) {
        return errorAt(loc, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    const auto newType = inType.pad(ShapeRef(padBegin.getValue()), ShapeRef(padEnd.getValue()));
    inferredReturnTypes.push_back(newType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::PadOp::serialize(EMU::BlobWriter& writer) {
    const auto padsBegin = writer.createVector(parseIntArrayAttr<uint32_t>(pads_begin_attr().getValue()));
    const auto padsEnd = writer.createVector(parseIntArrayAttr<uint32_t>(pads_end_attr().getValue()));

    MVCNN::PadParamsBuilder builder(writer);
    const auto padMode = vpux::VPUIP::convertVPUXPadMode2MVCNN(mode());
    builder.add_pad_mode(padMode);
    if (padMode == MVCNN::PadMode::PadMode_Constant) {
        builder.add_padValue(static_cast<float>(pad_value_attr()->convertToDouble()));
    }
    builder.add_pads_begin(padsBegin);
    builder.add_pads_end(padsEnd);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PadParams});
}
