//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::BucketizeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::BucketizeOpAdaptor bucketize(operands, attrs);
    if (mlir::failed(bucketize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = bucketize.data().getType().cast<NDTypeInterface>();
    const auto outType = inType.changeElemType(bucketize.output_type().getValue());
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::BucketizeOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::BucketizeParamsBuilder builder(writer);

    MVCNN::BucketizeOutputType outType;

    if (this->output_type().isSignedInteger(32)) {
        outType = MVCNN::BucketizeOutputType::BucketizeOutputType_I32;
    } else if (this->output_type().isSignedInteger(64)) {
        outType = MVCNN::BucketizeOutputType::BucketizeOutputType_I64;
    } else {
        VPUX_THROW("Unsupported type for output_type attribute. Got {0} instead of SI32 or SI64 type, that is defined "
                   "in schema for output_type.",
                   this->output_type());
    }

    builder.add_output_type(outType);
    builder.add_with_right_bound(with_right_bound());

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_BucketizeParams});
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(BucketizeOp op) {
    const mlir::Type constInt32 = getSInt32Type(op.getContext());
    const mlir::Type constInt64 = getSInt64Type(op.getContext());

    if (!(op.output_type() == constInt32 || op.output_type() == constInt64)) {
        return errorAt(op,
                       "Attribute output_type support only SI32 and SI64 type according to schema definition for this "
                       "attribute. Got {0} type",
                       op.output_type());
    }

    return mlir::success();
}
