//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/core/static_allocation.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

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
        auto type = val.getType().dyn_cast<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "StaticAllocation can work only with MemRef Type, got '{0}'", val.getType());

        return checked_cast<AddressType>(getTypeByteSize(type));
    }

    AddressType getAlignment(mlir::Value) const {
        return 64;
    }

    AddressType getAddress(mlir::Value val) const {
        auto addr = parent.getValOffset(val);
        VPUX_THROW_UNLESS(addr.hasValue(), "Value '{0}' was not allocated", val);

        return checked_cast<AddressType>(addr.getValue());
    }

    void allocated(mlir::Value val, AddressType addr) {
        VPUX_THROW_UNLESS(addr != InvalidAddress, "Trying to assign invalid address");
        VPUX_THROW_UNLESS(parent._valOffsets.count(val) == 0, "Value '{0}' was already allocated", val);

        parent._valOffsets.insert({val, checked_cast<int64_t>(addr)});
        parent._maxAllocatedSize = std::max(parent._maxAllocatedSize, alignVal<int64_t>(addr + getSize(val), 64));
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

vpux::StaticAllocation::StaticAllocation(mlir::Operation* rootOp) {
    LinearScan<mlir::Value, Handler> scan(std::numeric_limits<uint32_t>::max(), *this);

    auto callback = [&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::AllocOp>(op)) {
            auto val = allocOp.memref();

            VPUX_THROW_UNLESS(scan.alloc({val}, /*allowSpills*/ false),
                              "Failed to statically allocate memory with LinearScan for Value '{0}'", val);

            scan.handler().aliveValues.insert(val);
        } else if (auto deallocOp = mlir::dyn_cast<mlir::DeallocOp>(op)) {
            auto val = deallocOp.memref();

            scan.handler().aliveValues.erase(val);
            scan.freeNonAlive();
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
