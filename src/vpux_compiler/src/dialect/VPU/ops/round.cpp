//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RoundOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::RoundOpAdaptor round(operands, attrs);
    if (mlir::failed(round.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = round.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::RoundOp::serialize(EMU::BlobWriter& writer) {
    const auto roundMode = vpux::VPUIP::convertVPUXRoundMode2MVCNN(mode());
    const auto round = MVCNN::CreateRoundParams(writer, roundMode);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_RoundParams);
    builder.add_nested_params(round.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
