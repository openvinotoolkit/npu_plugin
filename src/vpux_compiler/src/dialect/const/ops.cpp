//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// ConstDialect::materializeConstant
//

mlir::Operation* vpux::Const::ConstDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType, mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// ConstDialect::populateBufferizePatterns
//

namespace {

class BufferizeConst final : public mlir::OpConversionPattern<Const::DeclareOp> {
public:
    BufferizeConst(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<Const::DeclareOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BufferizeConst::matchAndRewrite(Const::DeclareOp origOp, OpAdaptor,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, newType, origOp.contentAttr());
    return mlir::success();
}

}  // namespace

void vpux::Const::ConstDialect::populateBufferizePatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::TypeConverter& typeConverter, Logger log) {
    patterns.insert<BufferizeConst>(typeConverter, patterns.getContext(), log);
}

//
// DeclareOp::fold
//

mlir::OpFoldResult vpux::Const::DeclareOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.empty(), "constant has no operands");
    return contentAttr();
}

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyOp(Const::DeclareOp op) {
    const auto attrType = op.contentAttr().getType();
    const auto opType = op.getType().cast<vpux::NDTypeInterface>();

    if (opType.getShape() != attrType.getShape()) {
        return errorAt(op, "'Const.Declare' has mismatch in value shape '{0}' and result shape '{1}'",
                       attrType.getShape(), opType.getShape());
    }
    if (opType.getElementType() != attrType.getElementType()) {
        return errorAt(op, "'Const.Declare' has mismatch in value element type '{0}' and result element type '{1}'",
                       attrType.getElementType(), opType.getElementType());
    }

    const auto attrOrder = attrType.getDimsOrder();
    const auto opOrder = opType.getDimsOrder();

    if (opOrder != attrOrder) {
        return errorAt(op, "'Const.Declare' has mismatch in value DimsOrder '{0}' and result DimsOrder '{1}'",
                       attrOrder, opOrder);
    }

    return mlir::success();
}

}  // namespace

//
// setupExtraInterfaces
//

void Const::ConstDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addTypeInterface<Const::ConstDialect, mlir::RankedTensorType, vpux::TensorNDTypeInterface>();
    registry.addTypeInterface<Const::ConstDialect, mlir::UnrankedTensorType, vpux::TensorNDTypeInterface>();
    registry.addTypeInterface<Const::ConstDialect, mlir::MemRefType, vpux::MemRefNDTypeInterface>();
    registry.addTypeInterface<Const::ConstDialect, mlir::UnrankedMemRefType, vpux::MemRefNDTypeInterface>();
}

//
// Generated
//

#include <vpux/compiler/dialect/const/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/generated/ops.cpp.inc>
