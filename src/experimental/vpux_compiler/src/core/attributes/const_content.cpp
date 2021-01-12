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

#include "vpux/compiler/core/attributes/const_content.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// ConstContentRange
//

vpux::details::ConstContentBase::ConstContentBase(ArrayRef<char> data, bool isSplat, mlir::Type baseType,
                                                  ShapeRef shape, Optional<DimsOrder> actualDimsOrder)
        : _data(data), _isSplat(isSplat), _baseType(baseType), _shape(shape), _actualDimsOrder(actualDimsOrder) {
    VPUX_THROW_UNLESS(_baseType.isIntOrFloat(), "Only integers and floats are supported in ConstContentBase");
}

const char* vpux::details::ConstContentBase::getData(ptrdiff_t actualMemInd1D) const {
    VPUX_THROW_UNLESS(!_data.empty(), "NULL pointer dereference in ConstContent");

    if (_isSplat) {
        return _data.data();
    }

    const auto elemByteSize = _baseType.getIntOrFloatBitWidth() / CHAR_BIT;

    const auto baseDimsOrder = DimsOrder::fromNumDims(_shape.size());
    const auto actualDimsOrder = _actualDimsOrder.getValueOr(baseDimsOrder);

    if (actualDimsOrder == baseDimsOrder || actualDimsOrder == DimsOrder::fromNumDims(actualDimsOrder.numDims())) {
        const auto rawIndex = checked_cast<size_t>(actualMemInd1D * elemByteSize);
        VPUX_THROW_UNLESS(rawIndex < _data.size(), "Out-of-bound access in ConstContent");

        return _data.data() + rawIndex;
    }

    VPUX_THROW_UNLESS(actualDimsOrder.numDims() == baseDimsOrder.numDims(), "Can't reorder from '{0}' to '{1}'",
                      baseDimsOrder, actualDimsOrder);

    //
    // `actualMemInd1D` - 1D memory index for `actualDimsOrder`
    // We need to convert `actualMemInd1D` from 1D memory index to ND memory index,
    // then convert it to ND logical index and finally convert it to 1D logical index
    // to access the data.
    //

    const auto baseMemShape = baseDimsOrder.toMemoryOrder(_shape);
    const auto actualMemShape = actualDimsOrder.toMemoryOrder(_shape);

    const auto actualMemIndexND = getMemIndexND(actualMemInd1D, actualMemShape);
    const auto indexND = actualDimsOrder.toLogicalOrder(actualMemIndexND);
    const auto baseMemIndexND = baseDimsOrder.toMemoryOrder(indexND);
    const auto baseMemIndex1D = getMemIndex1D(baseMemIndexND, baseMemShape);

    const auto rawIndex = checked_cast<size_t>(baseMemIndex1D * elemByteSize);
    VPUX_THROW_UNLESS(rawIndex < _data.size(), "Out-of-bound access in ConstContent");

    return _data.data() + rawIndex;
}

//
// ConstContentAttr
//

bool vpux::ConstContentAttr::classof(mlir::Attribute attr) {
    if (attr.isa<mlir::DenseElementsAttr>()) {
        return true;
    }

    if (const auto opaque = attr.dyn_cast<mlir::OpaqueElementsAttr>()) {
        const size_t numElems = opaque.getNumElements();
        const size_t elemTypeByteSize = opaque.getType().getElementTypeBitWidth() / CHAR_BIT;

        const auto bytes = opaque.getValue();

        return bytes.size() == numElems * elemTypeByteSize;
    }

    return false;
}

namespace {

template <class Range>
void fillBuf(const Range& range, MutableArrayRef<char> buf) {
    using value_type = typename Range::iterator::value_type;
    static const auto VALUE_BYTE_SIZE = sizeof(value_type);

    VPUX_THROW_UNLESS(buf.size() == range.size() * VALUE_BYTE_SIZE,
                      "Buffer with byte size '{0}' is not enough to hold actual elements with '{1}' byte size",
                      buf.size(), range.size() * VALUE_BYTE_SIZE);

    loop_1d(LoopExecPolicy::Parallel, range.size(), [&](size_t i) {
        auto* bufPtr = reinterpret_cast<value_type*>(buf.data() + i * VALUE_BYTE_SIZE);
        *bufPtr = range[i];
    });
}

}  // namespace

void vpux::ConstContentAttr::convertTo(mlir::ShapedType actualType, MutableArrayRef<char> buf) const {
    VPUX_THROW_UNLESS(getNumElements() == actualType.getNumElements(),
                      "Content type '{0}' and actual type '{1}' are not compatible", getType(), actualType);

    const auto actualElemType = actualType.getElementType();

    Optional<DimsOrder> actualDimsOrder;
    if (const auto memref = actualType.dyn_cast<mlir::MemRefType>()) {
        const auto res = DimsOrder::fromType(memref);
        VPUX_THROW_UNLESS(res.hasValue(), "Can't get DimsOrder from Type '{0}'", memref);
        actualDimsOrder = res;
    }

    if (actualElemType.isUnsignedInteger(8)) {
        fillBuf(getValues<uint8_t>(actualDimsOrder), buf);
    } else if (actualElemType.isUnsignedInteger(16)) {
        fillBuf(getValues<uint16_t>(actualDimsOrder), buf);
    } else if (actualElemType.isUnsignedInteger(32)) {
        fillBuf(getValues<uint32_t>(actualDimsOrder), buf);
    } else if (actualElemType.isUnsignedInteger(64)) {
        fillBuf(getValues<uint64_t>(actualDimsOrder), buf);
    } else if (actualElemType.isSignedInteger(8)) {
        fillBuf(getValues<int8_t>(actualDimsOrder), buf);
    } else if (actualElemType.isSignedInteger(16)) {
        fillBuf(getValues<int16_t>(actualDimsOrder), buf);
    } else if (actualElemType.isSignedInteger(32)) {
        fillBuf(getValues<int32_t>(actualDimsOrder), buf);
    } else if (actualElemType.isSignedInteger(64)) {
        fillBuf(getValues<int64_t>(actualDimsOrder), buf);
    } else if (actualElemType.isF32()) {
        fillBuf(getValues<float>(actualDimsOrder), buf);
    } else if (actualElemType.isF64()) {
        fillBuf(getValues<double>(actualDimsOrder), buf);
    } else if (actualElemType.isF16()) {
        fillBuf(getValues<ngraph::float16>(actualDimsOrder), buf);
    } else if (actualElemType.isBF16()) {
        fillBuf(getValues<ngraph::bfloat16>(actualDimsOrder), buf);
    } else {
        VPUX_THROW("Unsupported element type '{0}'", actualElemType);
    }
}

bool vpux::ConstContentAttr::isSplat() const {
    if (const auto dense = dyn_cast<mlir::DenseElementsAttr>()) {
        return dense.isSplat();
    }

    bool isSplatBuffer = false;
    if (!mlir::DenseElementsAttr::isValidRawBuffer(getType(), getRawData(), isSplatBuffer)) {
        return false;
    }

    return isSplatBuffer;
}

ArrayRef<char> vpux::ConstContentAttr::getRawData() const {
    if (const auto dense = dyn_cast<mlir::DenseElementsAttr>()) {
        return dense.getRawData();
    }

    const auto opaque = cast<mlir::OpaqueElementsAttr>();
    const auto bytes = opaque.getValue();
    return makeArrayRef(bytes.data(), bytes.size());
}
