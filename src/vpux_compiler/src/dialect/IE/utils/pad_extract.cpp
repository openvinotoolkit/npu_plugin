//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::ArrayAttr padValue, Logger log) {
    if (padValue == nullptr) {
        log.nest().trace("Pad Attr is nullptr");
        return mlir::failure();
    }

    const auto valueVector = parseIntArrayAttr<int64_t>(padValue);

    if (valueVector.size() != 4) {
        log.nest().trace("Pad Attr size {1} != 4", valueVector.size());
        return mlir::failure();
    }

    return valueVector;
}

}  // namespace IE
}  // namespace vpux
