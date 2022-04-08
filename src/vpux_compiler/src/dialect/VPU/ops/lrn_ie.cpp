//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LRN_IEOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::LRN_IEOpAdaptor lrn_ie(operands, attrs);
    if (mlir::failed(lrn_ie.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lrn_ie.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::LRN_IEOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::String region;
    switch (this->region()) {
    case IE::LRN_IERegion::ACROSS:
        region = writer.createString("across");
        break;
    case IE::LRN_IERegion::SAME:
        region = writer.createString("same");
        break;
    default:
        VPUX_THROW("Unsupported LRN_IERegion {0}", this->region());
    }

    MVCNN::NormParamsBuilder builder(writer);
    builder.add_alpha(static_cast<float>(alpha().convertToDouble()));
    builder.add_beta(static_cast<float>(beta().convertToDouble()));
    builder.add_local_size(checked_cast<int32_t>(size()));
    builder.add_region(region);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormParams});
}
