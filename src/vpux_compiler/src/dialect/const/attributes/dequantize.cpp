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

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

//
// DequantizeAttr::inferOutputType
//

mlir::ShapedType vpux::Const::DequantizeAttr::inferOutputType(mlir::ShapedType input) const {
    const auto qElemType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(qElemType != nullptr, "Got non quantized type '{0}' in 'DequantizeAttr'");

    return input.clone(qElemType.getExpressedType());
}

//
// DequantizeAttr::transform
//

Const::Content vpux::Const::DequantizeAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                  mlir::Float32Type::get(getContext()), input.isSplat());

    const auto qElemType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>();
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

        const auto dimsOrder = DimsOrder::fromType(input.getType());
        const auto memAxis = dimsOrder.toMemDim(axis);
        const auto memShape = dimsOrder.toMemoryOrder(getShape(input.getType()));

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
