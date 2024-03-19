//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
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

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, newType, origOp.getContentAttr());
    return mlir::success();
}

}  // namespace

void vpux::Const::ConstDialect::populateBufferizePatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::TypeConverter& typeConverter, Logger log) {
    patterns.add<BufferizeConst>(typeConverter, patterns.getContext(), log);
}

//
// DeclareOp::fold
//

mlir::OpFoldResult vpux::Const::DeclareOp::fold(FoldAdaptor adaptor) {
    VPUX_THROW_UNLESS(adaptor.getOperands().empty(), "constant has no operands");
    return getContentAttr();
}

//
// DeclareOp::serialize
//

void vpux::Const::DeclareOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    vpux::Const::Content cnt = getContent();
    // int64_t typeTotalSize = cnt.getRawStorageBuf().size();

    auto tmpBuf = std::make_unique<char[]>(cnt.getType().getTotalAllocSize().count());

    MutableArrayRef<char> buf(tmpBuf.get(), cnt.getType().getTotalAllocSize().count());
    cnt.copyTo(buf);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(tmpBuf.get());
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

//
// DeclareOp::getBinarySize
//

size_t vpux::Const::DeclareOp::getBinarySize() {
    vpux::Const::Content cnt = getContent();

    return cnt.getType().getTotalAllocSize().count();
}

//
// DeclareOp::getAlignmentRequirements
//

size_t vpux::Const::DeclareOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}

//
// DeclareOp::getMemorySpace
//

vpux::VPURT::BufferSection vpux::Const::DeclareOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::Constant;
}

//
// DeclareOp::getAccessingProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getAccessingProcs() {
    auto tempFlagsVal = vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE;

    for (auto user : getResult().getUsers()) {
        if (auto binaryIface = mlir::dyn_cast<vpux::ELFNPU37XX::BinaryOpInterface>(user)) {
            tempFlagsVal = tempFlagsVal | binaryIface.getUserProcs();
        }
    }

    return tempFlagsVal;
}

//
// DeclareOp::getUserProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

//
// DeclareOp::verify
//

mlir::LogicalResult vpux::Const::DeclareOp::verify() {
    const auto op = getOperation();
    const auto attrType = getContentAttr().getType();
    const auto opType = getType().cast<vpux::NDTypeInterface>();
    // For type with swizzling skip the shape check as the content
    // might have been flattened to accomodate swizzled buffer.
    if (!vpux::getSwizzlingSchemeAttr(opType)) {
        if (opType.getShape() != attrType.getShape()) {
            return errorAt(op, "'Const.Declare' has mismatch in value shape '{0}' and result shape '{1}'",
                           attrType.getShape(), opType.getShape());
        }
    }
    if (opType.getElementType() != attrType.getElementType()) {
        if (!opType.getElementType().isa<mlir::quant::QuantizedType>() &&
            !attrType.getElementType().isa<mlir::IntegerType>()) {
            return errorAt(op, "'Const.Declare' has mismatch in value element type '{0}' and result element type '{1}'",
                           attrType.getElementType(), opType.getElementType());
        }
    }

    const auto attrOrder = attrType.getDimsOrder();
    const auto opOrder = opType.getDimsOrder();

    if (opOrder != attrOrder) {
        return errorAt(op, "'Const.Declare' has mismatch in value DimsOrder '{0}' and result DimsOrder '{1}'",
                       attrOrder, opOrder);
    }

    return mlir::success();
}

//
// setupExtraInterfaces
//

void Const::ConstDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        mlir::RankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::UnrankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::MemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
        mlir::UnrankedMemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/const/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/ops.cpp.inc>
