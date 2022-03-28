//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/compiler/conversion.hpp"

namespace vpux {

//
// LinearScanHandler
//

class LinearScanHandler final {
public:
    explicit LinearScanHandler(AddressType defaultAlignment = 1);

public:
    void markAsDead(mlir::Value val);
    void markAllBuffersAsDead();
    void markAsAlive(mlir::Value val);
    Byte maxAllocatedSize() const;
    bool checkInvariantExceedingNNCMX(mlir::Value val, vpux::AddressType baseOffset, vpux::AddressType cmxSize) const;

public:
    bool isAlive(mlir::Value val) const;
    static bool isFixedAlloc(mlir::Value val);
    static AddressType getSize(mlir::Value val);
    AddressType getAlignment(mlir::Value val) const;
    AddressType getAddress(mlir::Value val) const;
    void allocated(mlir::Value val, AddressType addr);
    void deallocate(mlir::Value val);
    mlir::DenseSet<mlir::Value> getAliveValues();
    void freed(mlir::Value val);
    static int getSpillWeight(mlir::Value);
    static bool spilled(mlir::Value);
    void setAddress(mlir::Value val, AddressType address);

private:
    IERT::SubViewOp getSubViewUserOp(mlir::Value val) const;
    bool hasEltwiseUser(mlir::Value val) const;
    vpux::AddressType calculateStaticOffsetWithStrides(ArrayRef<vpux::AddressType> subViewStaticOffsets,
                                                       StridesRef subViewStrides) const;
    bool addressWithStridesExceedsNNCMX(vpux::AddressType baseOffset, vpux::AddressType staticOffsetWithStrides,
                                        StridesRef subViewStrides, vpux::AddressType cmxSize) const;

private:
    mlir::DenseMap<mlir::Value, AddressType> _valOffsets;
    mlir::DenseSet<mlir::Value> _aliveValues;
    AddressType _defaultAlignment = 1;
    Byte _maxAllocatedSize;
};

}  // namespace vpux
