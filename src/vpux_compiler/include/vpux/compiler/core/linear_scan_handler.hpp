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

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseSet.h>

namespace vpux {

//
// LinearScanHandler
//

class LinearScanHandler final {
public:
    explicit LinearScanHandler(AddressType defaultAlignment = 1);

public:
    void markAsDead(mlir::Value val);
    void markAsAlive(mlir::Value val);
    Byte maxAllocatedSize() const;

public:
    bool isAlive(mlir::Value val) const;
    static bool isFixedAlloc(mlir::Value val);
    static AddressType getSize(mlir::Value val);
    AddressType getAlignment(mlir::Value val) const;
    AddressType getAddress(mlir::Value val) const;
    void allocated(mlir::Value val, AddressType addr);
    void deallocate(mlir::Value val);
    SmallVector<mlir::Value*> getIncreasingSizeOrderAlive();
    void freed(mlir::Value val);
    static int getSpillWeight(mlir::Value);
    static bool spilled(mlir::Value);
    void setAddress(mlir::Value val, AddressType address);

private:
    mlir::DenseMap<mlir::Value, AddressType> _valOffsets;
    mlir::DenseSet<mlir::Value> _aliveValues;
    AddressType _defaultAlignment = 1;
    Byte _maxAllocatedSize;
};

}  // namespace vpux
