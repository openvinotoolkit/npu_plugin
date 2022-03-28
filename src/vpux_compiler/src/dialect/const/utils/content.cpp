//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/utils/content.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/IE/loop.hpp"

using namespace vpux;

//
// Content::fromRawBuffer
//

Const::Content vpux::Const::Content::fromRawBuffer(vpux::NDTypeInterface type, ArrayRef<char> data,
                                                   mlir::Type storageElemType, bool isSplat) {
    Const::Content content;
    content._type = type;
    content._data = data;
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;
    return content;
}

//
// Content::allocTempBuffer
//

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat) {
    Const::Content content;
    content._type = type;
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;

    const size_t tempBufSize = isSplat ? 1 : checked_cast<size_t>(type.getNumElements());
    const Byte tempElemSize = vpux::getElemTypeSize(storageElemType);
    const size_t tempBufRawSize = tempBufSize * checked_cast<size_t>(tempElemSize.count());

    content._tempBuf.reset(new char[tempBufRawSize]);
    content._data = makeArrayRef(content._tempBuf.get(), tempBufRawSize);

    return content;
}

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat, size_t tempBufRawSize) {
    // Overloading for sub-byte cases.
    Const::Content content;
    content._type = type;
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;

    content._tempBuf.reset(new char[tempBufRawSize]);
    content._data = makeArrayRef(content._tempBuf.get(), tempBufRawSize);

    return content;
}

//
// Content::moveBuffer
//

Const::Content vpux::Const::Content::moveBuffer(vpux::NDTypeInterface type, Const::Content&& other) {
    Const::Content content;
    content._type = type;
    content._storageElemType = other._storageElemType;
    content._isSplat = other._isSplat;

    content._data = other._data;
    other._data = None;

    if (other._tempBuf != nullptr) {
        content._tempBuf = std::move(other._tempBuf);
    }

    return content;
}

//
// Content::copyTo
//

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

void vpux::Const::Content::copyTo(MutableArrayRef<char> buf) const {
    const Bit elemSize = vpux::getElemTypeSize(getElementType());
    const bool isTrivialStorage = (getElementType() == _storageElemType);
    const bool isSubByte = elemSize.count() < CHAR_BIT;
    // Dispatch is required when:
    // 1. The buffer is splat and expressed type doesn't match stored type (non-trivial).
    // 2. The buffer is splat and elements are not packed (fillBuf doesn't work after bitPack).
    // Otherwise, plain copy will do the trick.
    if (!_isSplat && (isTrivialStorage || isSubByte)) {
        VPUX_THROW_UNLESS(buf.size() == _data.size(),
                          "Byte sizes of the input buffer '{0}' and stored elements '{1}' are different.", buf.size(),
                          _data.size());
        std::memcpy(buf.data(), _data.data(), buf.size());
    } else {
        dispatchByElemType<void>(getElementType(), [this, buf](auto dummy) {
            using ElemT = std::decay_t<decltype(dummy)>;
            fillBuf(this->getValues<ElemT>(), buf);
        });
    }
}

//
// Content::fillWithZero
//

void vpux::Const::Content::fillWithZero() {
    if (auto perAxisQType = getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto outShape = getType().getShape();
        const auto order = getType().getDimsOrder();
        const auto outMemShape = order.toMemoryOrder(outShape);

        VPUX_THROW_UNLESS(outShape.size() == 4, "Unsupported shape size {0}", outShape.size());
        VPUX_THROW_UNLESS(perAxisQType.getQuantizedDimension() == 0, "Only per-channel quantization is supported");

        const auto OC = outShape[Dims4D::Filter::OC];
        const auto IC = outShape[Dims4D::Filter::IC];
        const auto H = outShape[Dims4D::Filter::KY];
        const auto W = outShape[Dims4D::Filter::KX];

        const auto zeroPoints = perAxisQType.getZeroPoints();
        for (int i = 0; i < OC; ++i) {
            const auto zp = zeroPoints[i];

            const auto fillChannel = [&](auto buffer) {
                loop_3d(LoopExecPolicy::Parallel, IC, H, W, [&](int64_t ic, int64_t h, int64_t w) {
                    using BufferType = std::decay_t<decltype(buffer)>;
                    using ElemType = typename BufferType::value_type;

                    const auto inMemIndND = order.toMemoryOrder(Shape{i, ic, h, w});
                    const auto inMemInd1D = getMemIndex1D(inMemIndND, outMemShape);

                    buffer[inMemInd1D] = checked_cast<ElemType>(zp);
                });
            };

            mutate(fillChannel);
        }

    } else if (auto qType = getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        const auto zp = qType.getZeroPoint();

        const auto fillBuffer = [&](auto buffer) {
            using BufferType = std::decay_t<decltype(buffer)>;
            using ElemType = typename BufferType::value_type;

            std::fill_n(buffer.data(), buffer.size(), checked_cast<ElemType>(zp));
        };

        mutate(fillBuffer);
    } else {
        auto outBuf = getRawTempBuf();
        std::fill_n(outBuf.data(), outBuf.size(), char(0));
    }
}
