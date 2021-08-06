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

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dim_values.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

#include <ie_layouts.h>

#include <functional>

namespace vpux {

constexpr size_t MAX_NAMED_ORDER_NUM_DIMS = 5;

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

    // Orders for 2D Convolution weights
    static const DimsOrder OIYX;
    static const DimsOrder OYXI;
    static const DimsOrder YXOI;

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

    bool isIdentity() const;

public:
    static DimsOrder fromPermutationAffineMap(mlir::AffineMap map);
    mlir::AffineMap toPermutationAffineMap(mlir::MLIRContext* ctx) const;

    static DimsOrder fromType(mlir::ShapedType type);
    static DimsOrder fromType(mlir::RankedTensorType type);
    static DimsOrder fromType(mlir::MemRefType type);

    static DimsOrder fromValue(mlir::Value val);

    SmallVector<mlir::AffineMap> toAffineMapsList(mlir::MLIRContext* ctx, ShapeRef shape) const;

public:
    bool isCompatibleLayout(mlir::MemRefType type) const;
    bool isCompatibleLayout(mlir::Value val) const;

public:
    static DimsOrder fromIE(InferenceEngine::Layout layout);
    InferenceEngine::Layout toIE() const;

public:
    Optional<StringLiteral> getCanonicalName() const;

public:
    template <typename T, template <class> class Tag>
    auto toMemoryOrder(details::DimValuesRef<Dim, T, Tag> values) const -> details::DimValues<MemDim, T, Tag> {
        VPUX_THROW_UNLESS(values.size() == numDims(), "DimValues '{0}' are not compatible with DimsOrder '{1}'", values,
                          *this);

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
        VPUX_THROW_UNLESS(values.size() == numDims(), "DimValues '{0}' are not compatible with DimsOrder '{1}'", values,
                          *this);

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
    explicit DimsOrder(StorageType code);

private:
    StorageType _code = 0;          // is serialized for the runtime
    StorageType _invertedCode = 0;  // to store permutation in major to minor order
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
