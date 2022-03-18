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

#include "vpux/compiler/core/linear_scan_handler.hpp"

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
    return val.getDefiningOp<IERT::StaticAllocOp>() != nullptr;
}

AddressType LinearScanHandler::getSize(mlir::Value val) {
    const Byte totalSize = getTotalSize(val);
    return checked_cast<AddressType>(totalSize.count());
}

AddressType LinearScanHandler::getAlignment(mlir::Value val) const {
    if (auto allocOp = val.getDefiningOp<mlir::memref::AllocOp>()) {
        if (auto alignment = allocOp.alignment()) {
            return checked_cast<AddressType>(alignment.getValue());
        }
    }

    return _defaultAlignment;
}

AddressType LinearScanHandler::getAddress(mlir::Value val) const {
    if (auto staticAllocOp = val.getDefiningOp<IERT::StaticAllocOp>()) {
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

    const auto endAddr = alignVal<int64_t>(addr + getSize(val), getAlignment(val));
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

IERT::SubViewOp LinearScanHandler::getSubViewUserOp(mlir::Value val) const {
    for (auto use : val.getUsers()) {
        if (mlir::isa<IERT::SubViewOp>(use)) {
            return mlir::cast<IERT::SubViewOp>(use);
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

vpux::AddressType LinearScanHandler::calculateStaticOffsetWithStrides(ArrayRef<vpux::AddressType> subViewStaticOffsets,
                                                                      StridesRef subViewStrides) const {
    Byte offset(0);

    for (auto p : zip(subViewStaticOffsets, subViewStrides)) {
        offset += Byte(std::get<0>(p) * std::get<1>(p));
    }

    return offset.count();
}

bool LinearScanHandler::addressWithStridesExceedsNNCMX(vpux::AddressType baseOffset,
                                                       vpux::AddressType staticOffsetWithStrides,
                                                       StridesRef subViewStrides, vpux::AddressType cmxSize) const {
    Byte cmxLeft(cmxSize - (baseOffset + staticOffsetWithStrides));

    for (auto stride : subViewStrides) {
        if (Byte(stride) > cmxLeft) {
            return true;
        }
    }

    return false;
}

bool LinearScanHandler::checkInvariantExceedingNNCMX(mlir::Value val, vpux::AddressType baseOffset,
                                                     vpux::AddressType cmxSize) const {
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

    const auto subViewStaticOffsets = parseIntArrayAttr<vpux::AddressType>(subView.static_offsets());
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