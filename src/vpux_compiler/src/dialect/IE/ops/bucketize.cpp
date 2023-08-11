//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::BucketizeOp::verify() {
    const mlir::Type constInt32 = getSInt32Type(getContext());
    const mlir::Type constInt64 = getSInt64Type(getContext());

    if (!(output_type() == constInt32 || output_type() == constInt64)) {
        return errorAt(*this,
                       "Attribute output_type support only SI32 and SI64 type according to schema definition for this "
                       "attribute. Got {0} type",
                       output_type());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::BucketizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BucketizeOpAdaptor bucketize(operands, attrs);
    if (mlir::failed(bucketize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = bucketize.data().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), bucketize.output_type());

    return mlir::success();
}
