// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

namespace {

template <class StorageType>
void getSparsityMap(Const::details::ContentRange<StorageType> inputBuffer, int64_t sparsifyValue,
                    llvm::MutableArrayRef<char> outBuffer) {
    for (size_t i = 0; i < outBuffer.size(); ++i) {
        for (size_t bitShift = 0; bitShift < CHAR_BIT; ++bitShift) {
            if (inputBuffer[i * CHAR_BIT + bitShift] != StorageType(sparsifyValue)) {
                outBuffer[i] |= (1 << bitShift);
            }
        }
    }
}

template <typename StorageType>
Const::Content generateSparsityMap(const Const::Content& content, int64_t sparsifyValue, NDTypeInterface outputType,
                                   mlir::MLIRContext* context) {
    const auto values = content.getValues<StorageType>();

    const auto sparsityMapElementType = mlir::IntegerType::get(context, 1, mlir::IntegerType::Unsigned);
    auto output = Const::Content::allocTempBuffer(outputType, sparsityMapElementType, false);
    output.fillWithZero();
    auto outBuf = output.getRawTempBuf();

    getSparsityMap(values, sparsifyValue, outBuf);

    return output;
}

}  // namespace

//
// GetSparsityMapAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::GetSparsityMapAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input.changeElemType(mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Unsigned));
}

//
// GetSparsityMapAttr::transform
//

Const::Content vpux::Const::GetSparsityMapAttr::transform(vpux::Const::Content& input) const {
    auto outputType = inferOutputType(input.getType());

    int64_t sparsifyValue = 0;
    auto inputElementType = input.getType().getElementType();
    if (auto qtype = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        inputElementType = normalizeQuantStorageType(qtype);
        sparsifyValue = qtype.getZeroPoint();
    }
    if (inputElementType.isSignedInteger(8)) {
        return generateSparsityMap<int8_t>(input, sparsifyValue, outputType, getContext());
    } else if (inputElementType.isUnsignedInteger(8)) {
        return generateSparsityMap<uint8_t>(input, sparsifyValue, outputType, getContext());
    } else if (inputElementType.isF16()) {
        return generateSparsityMap<float16>(input, sparsifyValue, outputType, getContext());
    } else if (inputElementType.isBF16()) {
        return generateSparsityMap<bfloat16>(input, sparsifyValue, outputType, getContext());
    } else if (inputElementType.isF32()) {
        return generateSparsityMap<float>(input, sparsifyValue, outputType, getContext());
    }
    VPUX_THROW("Unexpected weights data type: {0}", inputElementType);
}

//
// GetSparsityMapAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::GetSparsityMapAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::LAST;
}

//
// ContentAttr::getSparsityMap
//

Const::ContentAttr vpux::Const::ContentAttr::getSparsityMap() const {
    return get(*this, Const::GetSparsityMapAttr::get(getContext()).cast<Const::TransformAttrInterface>());
}
