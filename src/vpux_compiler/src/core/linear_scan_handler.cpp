//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

LinearScanHandler::LinearScanHandler(AddressType defaultAlignment): _defaultAlignment(defaultAlignment) {
}

void LinearScanHandler::markAsDead(mlir::Value val) {
    _aliveValues.erase(val);
}

void LinearScanHandler::markAllBuffersAsDead() {
    _aliveValues.clear();
}

void LinearScanHandler::markAsAlive(mlir::Value val) {
    _aliveValues.insert(val);
}

Byte LinearScanHandler::maxAllocatedSize() const {
    return _maxAllocatedSize;
}

bool LinearScanHandler::isAlive(mlir::Value val) const {
    return _aliveValues.contains(val);
}

bool LinearScanHandler::isFixedAlloc(mlir::Value val) {
    return val.getDefiningOp<VPUIP::StaticAllocOp>() != nullptr;
}

AddressType LinearScanHandler::getSize(mlir::Value val) {
    AddressType size;
    const auto it = _sizeCache.find(val);
    if (it == _sizeCache.end()) {
        size = getTotalSize(val).count();
        _sizeCache.insert({val, size});
    } else {
        size = it->second;
    }
    return size;
}

AddressType LinearScanHandler::getAlignment(mlir::Value val) const {
    if (auto allocOp = val.getDefiningOp<mlir::memref::AllocOp>()) {
        if (auto alignment = allocOp.getAlignment()) {
            return checked_cast<AddressType>(alignment.value());
        }
    } else if (auto allocOp = val.getDefiningOp<VPURT::Alloc>()) {
        if (auto alignment = allocOp.getAlignment()) {
            return checked_cast<AddressType>(alignment.value());
        }
    } else if (auto allocOp = val.getDefiningOp<VPURT::AllocDistributed>()) {
        if (auto alignment = allocOp.getAlignment()) {
            return checked_cast<AddressType>(alignment.value());
        }
    }

    return _defaultAlignment;
}

AddressType LinearScanHandler::getAddress(mlir::Value val) const {
    if (auto staticAllocOp = val.getDefiningOp<VPUIP::StaticAllocOp>()) {
        return checked_cast<AddressType>(staticAllocOp.offset());
    }

    const auto it = _valOffsets.find(val);
    VPUX_THROW_UNLESS(it != _valOffsets.end(), "Value '{0}' was not allocated", val);

    return it->second;
}

void LinearScanHandler::setAddress(mlir::Value val, AddressType address) {
    const auto it = _valOffsets.find(val);
    if (it == _valOffsets.end()) {
        _valOffsets.insert({val, address});
    } else {
        it->second = address;
    }
}

void LinearScanHandler::allocated(mlir::Value val, AddressType addr) {
    VPUX_THROW_UNLESS(addr != InvalidAddress, "Trying to assign invalid address");
    VPUX_THROW_UNLESS(_valOffsets.count(val) == 0, "Value '{0}' was already allocated", val);

    _valOffsets.insert({val, addr});

    const auto endAddr = alignValUp<int64_t>(addr + getSize(val), getAlignment(val));
    _maxAllocatedSize = Byte(std::max(_maxAllocatedSize.count(), endAddr));
}

void LinearScanHandler::deallocate(mlir::Value val) {
    VPUX_THROW_UNLESS(_valOffsets.count(val) > 0, "Value '{0}' was not allocated", val);

    _valOffsets.erase(val);
}

mlir::DenseSet<mlir::Value> LinearScanHandler::getAliveValues() {
    return _aliveValues;
}

void LinearScanHandler::freed(mlir::Value val) {
    markAsDead(val);
}

VPUIP::SubViewOp LinearScanHandler::getSubViewUserOp(mlir::Value val) const {
    for (auto use : val.getUsers()) {
        if (mlir::isa<VPUIP::SubViewOp>(use)) {
            return mlir::cast<VPUIP::SubViewOp>(use);
        }
    }

    return nullptr;
}

bool LinearScanHandler::hasEltwiseUser(mlir::Value val) const {
    for (auto use : val.getUsers()) {
        auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(use);
        if (nceClusterTaskOp && nceClusterTaskOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
            return true;
        }
    }

    return false;
}

AddressType LinearScanHandler::calculateStaticOffsetWithStrides(ArrayRef<AddressType> subViewStaticOffsets,
                                                                StridesRef subViewStrides) const {
    Byte offset(0);

    for (auto p : zip(subViewStaticOffsets, subViewStrides)) {
        offset += Byte(std::get<0>(p) * std::get<1>(p));
    }

    return offset.count();
}

bool LinearScanHandler::addressWithStridesExceedsNNCMX(AddressType baseOffset, AddressType staticOffsetWithStrides,
                                                       StridesRef subViewStrides, AddressType cmxSize) const {
    Byte cmxLeft(cmxSize - (baseOffset + staticOffsetWithStrides));

    for (auto stride : subViewStrides) {
        if (Byte(alignMemSize(stride, Byte(1))) > cmxLeft) {
            return true;
        }
    }

    return false;
}

bool LinearScanHandler::checkInvariantExceedingNNCMX(mlir::Value val, AddressType baseOffset,
                                                     AddressType cmxSize) const {
    // for concatenation in NNCMX with non contigious block memory write
    // prevent a scenario where tensor strides exceed NNCMX size

    auto subView = getSubViewUserOp(val);
    if (subView == nullptr) {
        return false;
    }

    if (!hasEltwiseUser(subView.result())) {
        // only impacted with Eltwise users
        return false;
    }

    const auto subViewStaticOffsets = parseIntArrayAttr<AddressType>(subView.static_offsets());
    const auto subViewStrides = getStrides(subView.source());
    VPUX_THROW_UNLESS(subViewStrides.size() == subViewStaticOffsets.size(),
                      "SubView offsets '{0}' doesn't match strides '{1}'", subViewStaticOffsets, subViewStrides);

    auto staticOffsetWithStrides = calculateStaticOffsetWithStrides(subViewStaticOffsets, subViewStrides);

    if (addressWithStridesExceedsNNCMX(baseOffset, staticOffsetWithStrides, subViewStrides, cmxSize)) {
        // set attribute to change the order of buffer allocation
        subView->setAttr("exceedingNNCMX", mlir::BoolAttr::get(subView.getContext(), true));
        return true;
    }

    return false;
}

int LinearScanHandler::getSpillWeight(mlir::Value) {
    VPUX_THROW("Spills are not implemented");
}

bool LinearScanHandler::spilled(mlir::Value) {
    VPUX_THROW("Spills are not implemented");
}
