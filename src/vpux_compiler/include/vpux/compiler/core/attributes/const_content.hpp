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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/data_convert.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
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

class ConstContentAttr final : public mlir::ElementsAttr {
public:
    using mlir::ElementsAttr::ElementsAttr;

public:
    static bool classof(mlir::Attribute attr);

public:
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

    auto getElemTypeSize() const {
        return vpux::getElemTypeSize(getType());
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

    template <typename T>
    auto getSplatValue() const {
        VPUX_THROW_UNLESS(isSplat(), "Expected the attribute to be a splat");
        return *getValues<T>().begin();
    }

public:
    ArrayRef<char> getRawData() const;
};

}  // namespace vpux
