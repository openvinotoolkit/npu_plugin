//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;
using namespace IE;

mlir::LogicalResult FloatOutConvRewriter::matchAndRewrite(IE::ConvolutionOp convolutionOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(convolutionOp) || !_isMixPrecisionSupported(convolutionOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(convolutionOp))) {
        return mlir::failure();
    }
    auto dequantizeInput = IE::findQuantizedInput(convolutionOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(convolutionOp.getFilter(), true);

    if (dequantizeInput == nullptr || filterDequantizeInput == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            convolutionOp, convolutionOp.getType(), dequantizeInput, filterDequantizeInput, convolutionOp.getBias(),
            convolutionOp.getStrides(), convolutionOp.getPadsBegin(), convolutionOp.getPadsEnd(),
            convolutionOp.getDilations(), convolutionOp.getPostOpAttr(), convolutionOp.getClampAttr());

    return mlir::success();
}

mlir::LogicalResult FloatOutGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(groupConvolutionOp) || !_isMixPrecisionSupported(groupConvolutionOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(groupConvolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeType = IE::findQuantizedInput(groupConvolutionOp.getInput(), true);
    auto filterDequantizeType = IE::findQuantizedInput(groupConvolutionOp.getFilter(), true);

    if (dequantizeType == nullptr || filterDequantizeType == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            groupConvolutionOp, groupConvolutionOp.getType(), dequantizeType, filterDequantizeType,
            groupConvolutionOp.getBias(), groupConvolutionOp.getStrides(), groupConvolutionOp.getPadsBegin(),
            groupConvolutionOp.getPadsEnd(), groupConvolutionOp.getDilations(), groupConvolutionOp.getGroupsAttr(),
            groupConvolutionOp.getPostOpAttr(), groupConvolutionOp.getClampAttr());

    return mlir::success();
}

mlir::LogicalResult FloatOutAddRewriter::matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(addOp) || !_isMixPrecisionSupported(addOp, false, _log)) {
        return mlir::failure();
    }
    // This transformation assumes that each input has IE::DequantizeOp producer
    auto lhsDequant = IE::findQuantizedInput(addOp.getInput1(), false);
    if (lhsDequant == nullptr) {
        return mlir::failure();
    }
    auto rhsDequant = IE::findQuantizedInput(addOp.getInput2(), false);
    if (rhsDequant == nullptr) {
        return mlir::failure();
    }

    // If target architecture does not support different scales, check that they are the same
    if (!_allowDifferentScales) {
        auto lhsType = lhsDequant.getType().cast<vpux::NDTypeInterface>();
        auto lhsQuantType = lhsType.getElementType().cast<mlir::quant::UniformQuantizedType>();

        auto rhsType = rhsDequant.getType().cast<vpux::NDTypeInterface>();
        auto rhsQuantType = rhsType.getElementType().cast<mlir::quant::UniformQuantizedType>();
        if (!isDoubleEqual(lhsQuantType.getScale(), rhsQuantType.getScale())) {
            return mlir::failure();
        }
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(addOp, addOp.getType(), lhsDequant, rhsDequant, addOp.getAutoBroadcast(),
                                           addOp.getPostOpAttr(), addOp.getClampAttr());

    return mlir::success();
}
