//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::CumSumOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::CumSumOpAdaptor cumSum(operands, attrs);
    if (mlir::failed(cumSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cumSum.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::CumSumOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::CumSumParamsBuilder builder(writer);
    builder.add_exclusive(checked_cast<bool>(exclusive().value_or(false)));
    builder.add_reverse(checked_cast<bool>(reverse().value_or(false)));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CumSumParams});
}
