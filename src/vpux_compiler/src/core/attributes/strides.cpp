//
// Copyright Intel Corporation.
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

#include "vpux/compiler/core/attributes/strides.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>

using namespace vpux;

//
// Strides utils
//

bool vpux::details::isDynamicDimValues(ArrayRef<Bit> strides) {
    return std::any_of(strides.begin(), strides.end(), [](Bit val) {
        return val.count() <= 0;
    });
}

//
// Strides
//

Strides vpux::getStrides(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getStrides();
}

//
// MemStrides
//

MemStrides vpux::getMemStrides(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getMemStrides();
}
