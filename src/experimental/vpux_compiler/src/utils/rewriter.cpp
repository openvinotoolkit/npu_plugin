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

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::rewriteFuncPrototype(mlir::FuncOp funcOp, mlir::TypeConverter& typeConverter,
                                               mlir::ConversionPatternRewriter& rewriter, Logger log) {
    log.trace("Update Function '@{0}' prototype", funcOp.sym_name());

    const auto funcType = funcOp.getType();

    mlir::TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
    for (const auto& p : funcType.getInputs() | indexed) {
        const auto newType = typeConverter.convertType(p.value());
        conversion.addInputs(checked_cast<uint32_t>(p.index()), newType);
    }

    SmallVector<mlir::Type, 1> newResultTypes;
    newResultTypes.reserve(funcOp.getNumResults());
    for (const auto& outType : funcType.getResults()) {
        newResultTypes.push_back(typeConverter.convertType(outType));
    }

    if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(), typeConverter, &conversion))) {
        return printTo(funcOp.emitError(), "Failed to convert Function arguments");
    }

    rewriter.updateRootInPlace(funcOp, [&]() {
        funcOp.setType(rewriter.getFunctionType(conversion.getConvertedTypes(), newResultTypes));
    });

    return mlir::success();
}
