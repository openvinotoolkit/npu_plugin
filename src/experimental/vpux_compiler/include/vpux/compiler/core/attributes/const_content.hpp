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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/data_convert.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

//
// ConstContentRange
//

namespace details {

class ConstContentBase {
public:
    ConstContentBase(ArrayRef<char> data, bool isSplat, mlir::Type baseType, ShapeRef shape,
                     Optional<DimsOrder> actualDimsOrder);

public:
    const char* getData(ptrdiff_t actualMemInd1D) const;

    mlir::Type baseType() const {
        return _baseType;
    }

public:
    bool operator==(const ConstContentBase& other) const {
        return _data == other._data && _actualDimsOrder == other._actualDimsOrder;
    }

    bool operator!=(const ConstContentBase& other) const {
        return !(*this == other);
    }

private:
    ArrayRef<char> _data;
    bool _isSplat = false;
    mlir::Type _baseType;
    ShapeRef _shape;
    Optional<DimsOrder> _actualDimsOrder;
};

template <typename T>
class ConstContentRange : public llvm::indexed_accessor_range<ConstContentRange<T>, ConstContentBase, T, T, T> {
    using BaseType = llvm::indexed_accessor_range<ConstContentRange<T>, ConstContentBase, T, T, T>;

public:
    ConstContentRange(ArrayRef<char> data, bool isSplat, mlir::Type baseType, ShapeRef shape,
                      Optional<DimsOrder> dimsOrder, ptrdiff_t count)
            : BaseType(ConstContentBase(data, isSplat, baseType, shape, dimsOrder), 0, count) {
    }

public:
    static T dereference(const ConstContentBase& base, ptrdiff_t index) {
        const auto* baseData = base.getData(index);
        return convertData<T>(baseData, base.baseType());
    }
};

}  // namespace details

//
// ConstContentAttr
//

class ConstContentAttr final : public mlir::Attribute {
public:
    using mlir::Attribute::Attribute;

public:
    static bool classof(mlir::Attribute attr);

public:
    mlir::ShapedType getType() const;

    auto getRank() const {
        return getType().getRank();
    }

    auto getShape() const {
        return vpux::getShape(getType());
    }

    auto getNumElements() const {
        return getType().getNumElements();
    }

    auto getElementType() const {
        return getType().getElementType();
    }

    auto getElementTypeBitWidth() const {
        return getType().getElementTypeBitWidth();
    }

    auto getSizeInBits() const {
        return getType().getSizeInBits();
    }

public:
    template <typename T>
    auto getValues(Optional<DimsOrder> order = None) const {
        return details::ConstContentRange<T>(getRawData(), isSplat(), getElementType(), getShape(), order,
                                             getNumElements());
    }

public:
    void convertTo(mlir::ShapedType actualType, MutableArrayRef<char> buf) const;

public:
    bool isSplat() const;

    mlir::Attribute getSplatValue() const {
        return getSplatDenseElements().getSplatValue();
    }

    template <typename T>
    auto getSplatValue() const {
        return getSplatDenseElements().getSplatValue<T>();
    }

public:
    ArrayRef<char> getRawData() const;

private:
    mlir::DenseElementsAttr getSplatDenseElements() const;
};

}  // namespace vpux
