//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

//
// DequantizeAttr::walkImmediateSubElements
//

void vpux::Const::DequantizeAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)>,
                                                           llvm::function_ref<void(mlir::Type)>) const {
}

//
// DequantizeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::DequantizeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const Bit typeSizeInBits = input.getElemTypeSize();
    VPUX_THROW_UNLESS(typeSizeInBits.count() >= CHAR_BIT, "Got sub-byte input '{0}' in DequantizeAttr",
                      input.getElementType());

    const auto qElemType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(qElemType != nullptr, "Got non quantized type '{0}' in 'DequantizeAttr'");

    return input.changeElemType(qElemType.getExpressedType());
}

//
// DequantizeAttr::transform
//

Const::Content vpux::Const::DequantizeAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                  mlir::Float32Type::get(getContext()), input.isSplat());

    const auto qElemType = input.getType().getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(qElemType != nullptr, "Got non quantized type '{0}' in 'DequantizeAttr'");

    const auto qVals = input.getValues<int64_t>();
    auto realVals = output.getTempBuf<float>();

    if (const auto uniformType = qElemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto scale = uniformType.getScale();
        const auto zeroPoint = uniformType.getZeroPoint();

        loop_1d(LoopExecPolicy::Parallel, realVals.size(), [&](size_t i) {
            realVals[i] = dequantize(qVals[i], scale, zeroPoint);
        });
    } else if (const auto uniformType = qElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto scales = uniformType.getScales();
        const auto zeroPoints = uniformType.getZeroPoints();
        const auto axis = Dim(uniformType.getQuantizedDimension());

        const auto dimsOrder = input.getType().getDimsOrder();
        const auto memAxis = dimsOrder.toMemDim(axis);
        const auto memShape = dimsOrder.toMemoryOrder(input.getType().getShape());

        const auto innerSize = memShape[memAxis];
        const auto outerSize = subspace::getTotalLines(memShape, memAxis);
        const auto outerShape = subspace::arrayElementExclude(memShape, memAxis);

        VPUX_THROW_UNLESS(scales.size() == checked_cast<size_t>(innerSize), "Wrong scales size '{0}', expected '{1}'",
                          scales.size(), innerSize);
        VPUX_THROW_UNLESS(zeroPoints.size() == checked_cast<size_t>(innerSize),
                          "Wrong zeroPoints size '{0}', expected '{1}'", zeroPoints.size(), innerSize);

        loop_2d(LoopExecPolicy::Parallel, outerSize, innerSize, [&](int64_t outerInd, int64_t innerInd) {
            const auto outerIndND = getMemIndexND(outerInd, outerShape);
            const auto fullIndND = subspace::arrayElementInclude(outerIndND, memAxis, innerInd);
            const auto fullInd1D = getMemIndex1D(fullIndND, memShape);

            realVals[fullInd1D] = dequantize(qVals[fullInd1D], scales[innerInd], zeroPoints[innerInd]);
        });
    } else {
        VPUX_THROW("Unsupported Quantized Type '{0}'", qElemType);
    }

    return output;
}

//
// ContentAttr::dequantize
//

Const::ContentAttr vpux::Const::ContentAttr::dequantize() const {
    return get(*this, Const::DequantizeAttr::get(getContext()).cast<Const::TransformAttrInterface>());
}
