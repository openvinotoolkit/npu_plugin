//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/prelu_pwl_table.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {

Optional<vpux::PWLTableEntry> findLeakyReluPWLEntry(const float reluSlope, const int64_t zeroPoint) {
    if (!isSupportedNegativeSlope(reluSlope)) {
        return None;
    }

    if (isFloatEqual(reluSlope, 0.0f)) {
        return getPWLEntryForAlpha0(zeroPoint);
    } else if (isFloatEqual(reluSlope, 0.1f)) {
        return getPWLEntryForAlpha1(zeroPoint);
    } else if (isFloatEqual(reluSlope, 0.2f)) {
        return getPWLEntryForAlpha2(zeroPoint);
    } else if (isFloatEqual(reluSlope, 0.25f)) {
        return getPWLEntryForAlpha25(zeroPoint);
    }

    return None;
}

Optional<vpux::PWLTableEntry> getLeakyReluPWLEntry(IE::PostOpAttr postOp, mlir::Type outElemType) {
    IE::LeakyReluOp::Adaptor leakyRelu(None, postOp.getAttrs());
    VPUX_THROW_UNLESS(leakyRelu.verify(mlir::UnknownLoc::get(postOp.getContext())).succeeded(),
                      "Wrong attributes '{0}' for '{1}' PostOp", postOp.getAttrs(), postOp.getName());

    const auto alpha = leakyRelu.negative_slope().convertToDouble();
    const auto zeroPoint = outElemType.cast<mlir::quant::UniformQuantizedType>().getZeroPoint();
    return findLeakyReluPWLEntry(static_cast<float>(alpha), zeroPoint);
}

Optional<vpux::PWLTableEntry> findCustomPWLTable(IE::PostOpAttr postOp, mlir::Type outElemType) {
    // create map
    // this create map is a temporary solution, it will be change in a future MR when we will decide if we will add
    // custom tables and compilation train tables to MLIR or an analysis.

    if (!outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        return None;
    }

    using PWLEntryFunc = Optional<vpux::PWLTableEntry> (*)(IE::PostOpAttr, mlir::Type);
    std::map<std::string, PWLEntryFunc> pwlTableMap = {
            {"IE.LeakyRelu", getLeakyReluPWLEntry},
    };

    const StringRef activationName = postOp.getName().getValue();
    auto pwlTableIt = pwlTableMap.find(activationName.str());
    if (pwlTableIt == pwlTableMap.end()) {
        return None;
    }
    return pwlTableMap[activationName.str()](postOp, outElemType);
}

bool isSupportedNegativeSlope(const float reluSlope) {
    const std::vector<float> supportedSlopes = {0.0f, 0.1f, 0.2f, 0.25f};
    const auto slopePredicate = [reluSlope](const float supportedSlope) -> bool {
        // The reason why we used isFloatEqual() is because PRelu has only FP16 and FP32 precision for
        // negative_slope. We need to compare them in float precision, and because we convert PRelu to LeakyRelu
        // negative_slope is in double and it is possible the double epsilon to be smaller than the subtraction
        // result.
        return isFloatEqual(reluSlope, supportedSlope);
    };
    return std::any_of(supportedSlopes.cbegin(), supportedSlopes.cend(), slopePredicate);
}

bool isSupportedPReLU(const float reluSlope, const int64_t zeroPoint) {
    const auto maybePReluPwl = findLeakyReluPWLEntry(reluSlope, zeroPoint);
    return maybePReluPwl.has_value();
}

}  // namespace vpux
