//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/core/static_allocation.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

using namespace vpux;

struct vpux::StaticAllocation::Handler final {
    StaticAllocation& parent;
    mlir::DenseSet<mlir::Value> aliveValues;

    explicit Handler(StaticAllocation& parent): parent(parent) {
    }

    bool isAlive(mlir::Value val) const {
        return aliveValues.contains(val);
    }

    bool isFixedAlloc(mlir::Value) const {
        return false;
    }

    AddressType getSize(mlir::Value val) const {
        const auto type = val.getType().dyn_cast<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "StaticAllocation can work only with MemRef Type, got '{0}'", val.getType());

        const Byte totalSize = getTypeTotalSize(type);
        return checked_cast<AddressType>(totalSize.count());
    }

    AddressType getAlignment(mlir::Value) const {
        return parent._alignment;
    }

    AddressType getAddress(mlir::Value val) const {
        const auto addr = parent.getValOffset(val);
        VPUX_THROW_UNLESS(addr.hasValue(), "Value '{0}' was not allocated", val);

        return checked_cast<AddressType>(addr.getValue());
    }

    void allocated(mlir::Value val, AddressType addr) {
        VPUX_THROW_UNLESS(addr != InvalidAddress, "Trying to assign invalid address");
        VPUX_THROW_UNLESS(parent._valOffsets.count(val) == 0, "Value '{0}' was already allocated", val);

        parent._valOffsets.insert({val, checked_cast<int64_t>(addr)});

        const auto endAddr = alignVal<int64_t>(addr + getSize(val), getAlignment(val));
        parent._maxAllocatedSize = Byte(std::max(parent._maxAllocatedSize.count(), endAddr));
    }

    void freed(mlir::Value) const {
    }

    int getSpillWeight(mlir::Value) const {
        VPUX_THROW("Spills is not allowed");
    }

    bool spilled(mlir::Value) const {
        VPUX_THROW("Spills is not allowed");
    }
};

vpux::StaticAllocation::StaticAllocation(mlir::Operation* rootOp, mlir::Attribute memSpace, Byte maxSize,
                                         uint64_t alignment)
        : _alignment(alignment) {
    LinearScan<mlir::Value, Handler> scan(maxSize.count(), *this);

    auto callback = [&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto val = allocOp.memref();
            const auto type = val.getType().dyn_cast<mlir::MemRefType>();

            if (type == nullptr) {
                return;
            }
            if (type.getMemorySpace() != memSpace) {
                return;
            }

            VPUX_THROW_UNLESS(scan.alloc({val}, /*allowSpills*/ false),
                              "Failed to statically allocate memory with LinearScan for Value '{0}'", val);

            scan.handler().aliveValues.insert(val);
        } else if (auto deallocOp = mlir::dyn_cast<mlir::memref::DeallocOp>(op)) {
            const auto val = deallocOp.memref();

            if (scan.handler().aliveValues.erase(val)) {
                scan.freeNonAlive();
            }
        }
    };

    rootOp->walk(callback);
}

Optional<int64_t> vpux::StaticAllocation::getValOffset(mlir::Value val) const {
    auto it = _valOffsets.find(val);
    if (it != _valOffsets.end()) {
        return it->second;
    }
    return None;
}
