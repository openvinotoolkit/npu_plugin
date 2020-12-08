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

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dim_values.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>

#include <ie_layouts.h>

#include <functional>

namespace vpux {

//
// DimsOrder
//

class DimsOrder final {
public:
    using StorageType = uint64_t;

    static constexpr size_t BITS_PER_DIM = 4;
    static constexpr size_t MAX_DIM_IND = (1 << BITS_PER_DIM) - 1;

    static_assert(sizeof(StorageType) * 8 / BITS_PER_DIM >= MAX_NUM_DIMS,
                  "StorageType is not enough to hold MAX_NUM_DIMS dimensions");
    static_assert(MAX_DIM_IND >= MAX_NUM_DIMS, "StorageType is not enough to hold MAX_NUM_DIMS dimensions");

public:
    static const DimsOrder C;
    static const DimsOrder NC;
    static const DimsOrder CHW;
    static const DimsOrder HWC;
    static const DimsOrder HCW;
    static const DimsOrder NCHW;
    static const DimsOrder NHWC;
    static const DimsOrder NHCW;
    static const DimsOrder NCDHW;
    static const DimsOrder NDHWC;

public:
    static void validateCode(StorageType code);
    static void validateNumDims(size_t numDims);
    static void validatePermutation(DimArrRef perm);

public:
    static StorageType getCodeFromNumDims(size_t numDims);
    static StorageType getCodeFromPermutation(DimArrRef perm);

public:
    DimsOrder() = default;

    static DimsOrder fromCode(StorageType code);
    static DimsOrder fromNumDims(size_t numDims);
    static DimsOrder fromPermutation(DimArrRef perm);

public:
    StorageType code() const {
        return _code;
    }

public:
    bool empty() const {
        return _code == 0;
    }

    size_t numDims() const;

public:
    bool hasDim(Dim d) const;

    size_t dimPos(Dim d) const;

    Dim dimAt(size_t pos) const;

public:
    MemDim toMemDim(Dim d) const;
    Dim toDim(MemDim d) const;

public:
    // Convert from packed format to array of dimensions from major to minor.
    DimArr toPermutation() const;

public:
    static Optional<DimsOrder> fromAffineMap(mlir::AffineMap map);
    mlir::AffineMap toAffineMap(mlir::MLIRContext* ctx) const;

    static Optional<DimsOrder> fromType(mlir::MemRefType type);

public:
    static DimsOrder fromIE(InferenceEngine::Layout layout);
    InferenceEngine::Layout toIE() const;

public:
    Optional<StringRef> getCanonicalName() const;

public:
    template <typename T, template <class> class Tag>
    auto toMemoryOrder(details::DimValuesRef<Dim, T, Tag> values) const -> details::DimValues<MemDim, T, Tag> {
        assert(values.size() == numDims());

        return to_container<details::DimValues<MemDim, T, Tag>>(toPermutation() | transformed([values](Dim d) {
                                                                    return values[d];
                                                                }));
    }
    template <typename T, template <class> class Tag>
    auto toMemoryOrder(const details::DimValues<Dim, T, Tag>& values) const {
        return toMemoryOrder(details::DimValuesRef<Dim, T, Tag>(values));
    }

    template <typename T, template <class> class Tag>
    auto toLogicalOrder(details::DimValuesRef<MemDim, T, Tag> values) const -> details::DimValues<Dim, T, Tag> {
        assert(values.size() == numDims());

        return to_container<details::DimValues<Dim, T, Tag>>(irange(values.size()) |
                                                             transformed([this, values](size_t dimInd) {
                                                                 const auto dim = Dim(dimInd);
                                                                 const auto memDim = this->toMemDim(dim);
                                                                 return values[memDim];
                                                             }));
    }
    template <typename T, template <class> class Tag>
    auto toLogicalOrder(const details::DimValues<MemDim, T, Tag>& values) const {
        return toLogicalOrder(details::DimValuesRef<MemDim, T, Tag>(values));
    }

public:
    void printFormat(llvm::raw_ostream& streams) const;

private:
    explicit DimsOrder(StorageType code): _code(code) {
    }

private:
    StorageType _code = 0;
};

inline bool operator==(DimsOrder order1, DimsOrder order2) {
    return order1.code() == order2.code();
}
inline bool operator!=(DimsOrder order1, DimsOrder order2) {
    return order1.code() != order2.code();
}

}  // namespace vpux

//
// Hash
//

namespace std {

template <>
struct hash<vpux::DimsOrder> final {
    size_t operator()(vpux::DimsOrder order) const {
        return order.code();
    }
};

}  // namespace std
