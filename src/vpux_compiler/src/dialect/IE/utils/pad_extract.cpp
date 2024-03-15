//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/utils/error.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::ArrayAttr padValue, Logger log) {
    if (padValue == nullptr) {
        log.nest().trace("Pad Attr is nullptr");
        return mlir::failure();
    }

    return parseIntArrayAttr<int64_t>(padValue);
}

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::Location loc, const mlir::Value& padValue,
                                                  const std::optional<mlir::ArrayAttr>& padAttr,
                                                  vpux::ShapeRef inputShape) {
    if (padAttr.has_value()) {
        return parseIntArrayAttr<int64_t>(padAttr.value());
    } else if (padValue != nullptr) {
        auto padsConst = padValue.getDefiningOp<Const::DeclareOp>();
        if (padsConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for pad");
        }

        auto padValueShape = padValue.getType().cast<vpux::NDTypeInterface>().getShape().raw();
        if (padValueShape.size() != 1 || padValueShape[0] != checked_cast<int64_t>(inputShape.size())) {
            return errorAt(loc, "pad_begin shape is not compatible with input tensor."
                                "The length of the list must be equal to the number of dimensions in the input tensor");
        }

        const auto padContent = padsConst.getContent();
        return to_small_vector(padContent.getValues<int64_t>());
    }

    return errorAt(loc, "Pads were not provided");
}

}  // namespace IE
}  // namespace vpux
