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

Strides vpux::getStrides(mlir::RankedTensorType type) {
    const auto memStrides = getMemStrides(type);
    const auto order = DimsOrder::fromType(type);
    return order.toLogicalOrder(memStrides);
}

Strides vpux::getStrides(mlir::MemRefType type) {
    const auto maps = type.getAffineMaps();

    mlir::MemRefType canonicalType;
    if (maps.size() <= 1) {
        canonicalType = type;
    } else {
        auto composedMap = maps.back();

        for (auto map : maps.drop_back() | reversed) {
            composedMap = composedMap.compose(map);
        }

        canonicalType = mlir::MemRefType::Builder(type).setAffineMaps({composedMap});
    }

    SmallVector<int64_t> elemStrides;
    int64_t offset = 0;
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::getStridesAndOffset(canonicalType, elemStrides, offset)),
                      "Only strided/simple MemRef Types are supported, got '{0}'", type);
    VPUX_THROW_UNLESS(elemStrides.size() == checked_cast<size_t>(type.getRank()),
                      "Only strided/simple MemRef Types are supported, got '{0}'", type);
    VPUX_THROW_UNLESS(offset == 0, "MemRef with non-zero offset '{0}' is not supported", offset);

    const auto elemSize = getElemTypeSize(type);
    Strides strides(elemStrides.size());

    for (auto i : irange(strides.size())) {
        strides[Dim(i)] = elemSize * elemStrides[i];
    }

    return strides;
}

Strides vpux::getStrides(mlir::ShapedType type) {
    return llvm::TypeSwitch<mlir::ShapedType, Strides>(type)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return getStrides(memref);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return getStrides(tensor);
            })
            .Default([](mlir::ShapedType type) -> Strides {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

Strides vpux::getStrides(mlir::Value val) {
    const auto type = val.getType().dyn_cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getStrides(type);
}

//
// MemStrides
//

MemStrides vpux::getMemStrides(mlir::RankedTensorType type) {
    const auto order = DimsOrder::fromType(type);

    // Tensors are always compact
    const auto reqs = StrideReqs::compact(checked_cast<size_t>(type.getRank()));
    return reqs.calcStrides(order, type);
}

MemStrides vpux::getMemStrides(mlir::MemRefType type) {
    const auto order = DimsOrder::fromType(type);
    const auto strides = getStrides(type);
    return order.toMemoryOrder(strides);
}

MemStrides vpux::getMemStrides(mlir::ShapedType type) {
    return llvm::TypeSwitch<mlir::ShapedType, MemStrides>(type)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return getMemStrides(memref);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return getMemStrides(tensor);
            })
            .Default([](mlir::ShapedType type) -> MemStrides {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

MemStrides vpux::getMemStrides(mlir::Value val) {
    const auto type = val.getType().dyn_cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getMemStrides(type);
}
