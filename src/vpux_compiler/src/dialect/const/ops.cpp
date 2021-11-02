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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

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
// DeclareOp::serialize
//

void vpux::Const::DeclareOp::serialize(std::vector<char>& buffer) {
    /*
    // From build-x86_64_cached/src/vpux_compiler/include/vpux/compiler/dialect/const/generated/ops.hpp.inc:
    // vpux::Const::Content content();

    // Defined in include/vpux/compiler/dialect/const/utils/content.hpp
    details::ContentRange<OutT> = cntnt.getValues();
    */

    //(void)buffer;
    /*
    for (int i = 0; i < 32; i++) {
        buffer.push_back(i + 128);
    }
    */

    // <<error: cannot bind non-const lvalue reference of type ‘vpux::Const::Content&’ to an rvalue of type
    // ‘vpux::Const::Content’>> vpux::Const::Content& cnt = content();
    vpux::Const::Content cnt = content();

    /*
    // Does NOT work well for conversions from content with e.g. f16 values to
    //   char - it just returns 0 bytes.
    const auto range = cnt.getValues<char>();
    for (char val : range) {
        printf("val = %x\n", val);
    }
    fflush(stdout);
    */

    printf("content().getNumElements() = %ld\n", cnt.getNumElements());
    // const vpux::MemType::Bit tmpBit = cnt.getElemTypeSize();
    const Byte aByte = cnt.getElemTypeSize();
    // uint32_t tmp = static_cast<uint32_t>(tmpBit);
    int64_t elemTypeSize = aByte.count();
    printf("content().getElemTypeSize() = %ld\n", elemTypeSize);
    //
    // printf("content().getTypeTotalSize() = %ld\n", cnt.getTypeTotalSize());
    // Inspired from vpux_compiler/src/dialect/const/attributes/reorder.cpp
    // Note: Byte seems to be defined in src/vpux_utils/include/vpux/utils/core/mem_size.hpp, in enum class MemType
    const Byte aByteTotal = cnt.getTypeTotalSize();
    int64_t typeTotalSize = aByteTotal.count();
    printf("content().getTypeTotalSize() = %ld\n", typeTotalSize);
    fflush(stdout);

    // See https://llvm.org/doxygen/classllvm_1_1ArrayRef.html
    // Gives <<error: use of deleted function>>: llvm::ArrayRef<char> tmp = content().getRawStorageBuf();

    // char* tmpBuf = (char*)malloc(typeTotalSize * sizeof(char));
    char* tmpBuf = new char[typeTotalSize];

    // From https://llvm.org/doxygen/classllvm_1_1MutableArrayRef.html: "This is intended to be trivially copyable, so
    // it should be passed by value."
    // MutableArrayRef<char> buf;
    MutableArrayRef<char> buf(tmpBuf, typeTotalSize);
    // See vpux_compiler/src/dialect/const/utils/content.cpp
    // content().copyTo(buf);
    cnt.copyTo(buf);

    for (std::size_t i = 0; i < buf.size(); i++) {
        buffer.push_back(buf[i]);
    }

    // free(tmpBuf);
    delete tmpBuf;
}

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyOp(Const::DeclareOp op) {
    const auto attrType = op.contentAttr().getType();
    const auto opType = op.getType().cast<mlir::ShapedType>();

    if (opType.getShape() != attrType.getShape()) {
        return errorAt(op, "'Const.Declare' has mismatch in value shape '{0}' and result shape '{1}'",
                       attrType.getShape(), opType.getShape());
    }
    if (opType.getElementType() != attrType.getElementType()) {
        return errorAt(op, "'Const.Declare' has mismatch in value element type '{0}' and result element type '{1}'",
                       attrType.getElementType(), opType.getElementType());
    }

    const auto attrOrder = DimsOrder::fromType(attrType);
    const auto opOrder = DimsOrder::fromType(opType);

    if (opOrder != attrOrder) {
        return errorAt(op, "'Const.Declare' has mismatch in value DimsOrder '{0}' and result DimsOrder '{1}'",
                       attrOrder, opOrder);
    }

    return mlir::success();
}

}  // namespace

//
// Generated
//

#include <vpux/compiler/dialect/const/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/generated/ops.cpp.inc>
