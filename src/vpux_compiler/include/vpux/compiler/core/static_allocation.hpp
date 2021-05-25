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

#pragma once

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

class StaticAllocation final {
public:
    explicit StaticAllocation(mlir::Operation* rootOp, mlir::Attribute memSpace = nullptr,
                              Byte maxSize = Byte(std::numeric_limits<uint64_t>::max()), uint64_t alignment = 1);

public:
    Optional<int64_t> getValOffset(mlir::Value val) const;

    auto maxAllocatedSize() const {
        return _maxAllocatedSize;
    }

private:
    struct Handler;
    friend Handler;

private:
    uint64_t _alignment = 1;
    mlir::DenseMap<mlir::Value, int64_t> _valOffsets;
    Byte _maxAllocatedSize;
};

}  // namespace vpux
