//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExpandOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExpandOpAdaptor expand(operands, attrs);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(expand.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(expand.pads_end());

    const auto inType = expand.input().getType().cast<vpux::NDTypeInterface>();

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    inferredReturnTypes.push_back(newType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ExpandOp::serialize(EMU::BlobWriter& writer) {
    const auto padsBegin = writer.createVector(parseIntArrayAttr<uint32_t>(pads_begin()));
    const auto padsEnd = writer.createVector(parseIntArrayAttr<uint32_t>(pads_end()));

    MVCNN::PadParamsBuilder builder(writer);
    builder.add_pad_mode(MVCNN::PadMode::PadMode_Constant);
    builder.add_padValue(0.0);
    builder.add_pads_begin(padsBegin);
    builder.add_pads_end(padsEnd);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PadParams});
}
