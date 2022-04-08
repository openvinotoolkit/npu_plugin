// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <numeric>

using namespace vpux;

namespace {

template <typename StorageType>
Const::Content sparsify(const Const::Content& content, int64_t sparsifyValue, NDTypeInterface outputType) {
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), false);
    output.fillWithZero();
    auto outBuf = output.getRawTempBuf();
    auto outBlobPtr = reinterpret_cast<StorageType*>(outBuf.data());

    auto inputValues = content.getValues<StorageType>();

    auto shape = content.getShape();
    VPUX_THROW_UNLESS(shape.size() == 4, "Const::Content::sparsify: got unxpected content shape {0}", shape.size());

    const auto OC = shape[vpux::Dims4D::Filter::OC];
    const auto IC = shape[vpux::Dims4D::Filter::IC];
    const auto KY = shape[vpux::Dims4D::Filter::KY];
    const auto KX = shape[vpux::Dims4D::Filter::KX];
    const auto workloadSize = IC * KY * KX;
    for (int64_t oc = 0; oc < OC; ++oc) {
        auto begin = oc * workloadSize;
        auto end = (oc + 1) * workloadSize;
        for (auto inputIndex = begin, outputIndex = begin; inputIndex < end; ++inputIndex) {
            const auto inputValue = inputValues[inputIndex];
            if (inputValue == checked_cast<StorageType>(sparsifyValue)) {
                continue;
            }
            outBlobPtr[outputIndex++] = inputValue;
        }
    }
    return output;
}

}  // namespace

//
// SparsifyAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::SparsifyAttr::inferOutputType(vpux::NDTypeInterface input) const {
    VPUX_THROW_UNLESS(input.getDimsOrder() == DimsOrder::OIYX || input.getDimsOrder() == DimsOrder::OYXI,
                      "SparsifyAttr: Unsupported DimsOrder");
    // TODO: In that case the size of sparsified constant will be the same as dense case, which is not, what we want.
    // Actually, we can't infer output size from input type only, it is data dependent. Looks like we can't implement
    // sparsify operation on top of constant folding mechanism
    // [Track number: E#24341].
    return input;
}

//
// SparsifyAttr::transform
//

Const::Content vpux::Const::SparsifyAttr::transform(vpux::Const::Content& input) const {
    auto outputType = inferOutputType(input.getType());

    int64_t sparsifyValue = 0;
    auto inputElementType = input.getType().getElementType();
    if (auto qtype = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        inputElementType = normalizeQuantStorageType(qtype);
        sparsifyValue = qtype.getZeroPoint();
    }
    if (inputElementType.isSignedInteger(8)) {
        return sparsify<int8_t>(input, sparsifyValue, outputType);
    } else if (inputElementType.isUnsignedInteger(8)) {
        return sparsify<uint8_t>(input, sparsifyValue, outputType);
    } else if (inputElementType.isF16()) {
        return sparsify<float16>(input, sparsifyValue, outputType);
    } else if (inputElementType.isBF16()) {
        return sparsify<bfloat16>(input, sparsifyValue, outputType);
    } else if (inputElementType.isF32()) {
        return sparsify<float>(input, sparsifyValue, outputType);
    }
    VPUX_THROW("Unexpected weights data type: {0}", inputElementType);
}

//
// ContentAttr::sparsify
//

Const::ContentAttr vpux::Const::ContentAttr::sparsify() const {
    return get(*this, Const::SparsifyAttr::get(getContext()).cast<Const::TransformAttrInterface>());
}
