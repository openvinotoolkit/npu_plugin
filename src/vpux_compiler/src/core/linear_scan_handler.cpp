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
    const auto type = val.getType().dyn_cast<mlir::MemRefType>();
    VPUX_THROW_UNLESS(type != nullptr, "StaticAllocation can work only with MemRef Type, got '{0}'", val.getType());

    const Byte totalSize = getTotalSize(type);
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

SmallVector<mlir::Value> LinearScanHandler::getIncreasingSizeOrderAlive() {
    SmallVector<std::pair<mlir::Value, AddressType>> orderBuffers;
    for (auto& alive : _aliveValues) {
        AddressType size = getSize(alive);
        orderBuffers.push_back(std::make_pair(alive, size));
    }
    llvm::sort(orderBuffers.begin(), orderBuffers.end(),
               [](const std::pair<mlir::Value, AddressType>& val1, const std::pair<mlir::Value, AddressType>& val2) {
                   if (val1.second == val2.second) {
                       if (const auto loc1 = val1.first.getLoc().dyn_cast<mlir::NameLoc>()) {
                           if (const auto loc2 = val2.first.getLoc().dyn_cast<mlir::NameLoc>()) {
                               StringRef name1 = loc1.getName().strref();
                               StringRef name2 = loc2.getName().strref();
                               return name1.compare(name2) < 0;
                           }
                       }
                   }
                   return val1.second < val2.second;
               });
    SmallVector<mlir::Value> orderedBuffers;
    for (auto& buf : orderBuffers) {
        orderedBuffers.push_back(buf.first);
    }
    return orderedBuffers;
}

void LinearScanHandler::freed(mlir::Value val) {
    markAsDead(val);
}

int LinearScanHandler::getSpillWeight(mlir::Value) {
    VPUX_THROW("Spills are not implemented");
}

bool LinearScanHandler::spilled(mlir::Value) {
    VPUX_THROW("Spills are not implemented");
}