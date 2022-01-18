//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
