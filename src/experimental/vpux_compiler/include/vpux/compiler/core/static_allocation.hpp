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
