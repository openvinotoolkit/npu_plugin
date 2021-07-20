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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/DecodeAttributesInterfaces.h>

using namespace vpux;

namespace {

//
// IEAsmHooks
//

class IEAsmHooks final : public mlir::OpAsmDialectInterface {
public:
    using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

public:
    mlir::LogicalResult getAlias(mlir::Attribute attr, llvm::raw_ostream& os) const final;
};

mlir::LogicalResult IEAsmHooks::getAlias(mlir::Attribute attr, llvm::raw_ostream& os) const {
    if (const auto mapAttr = attr.dyn_cast<mlir::AffineMapAttr>()) {
        const auto map = mapAttr.getValue();

        if (map.isPermutation()) {
            const auto dimsOrder = DimsOrder::fromPermutationAffineMap(map);

            if (const auto name = dimsOrder.getCanonicalName()) {
                os << name.getValue();
                return mlir::success();
            }
        }
    }

    return mlir::failure();
}

//
// IEDecodeAttributesHooks
//

class IEDecodeAttributesHooks final : public mlir::DialectDecodeAttributesInterface {
public:
    using mlir::DialectDecodeAttributesInterface::DialectDecodeAttributesInterface;

public:
    mlir::LogicalResult decode(mlir::OpaqueElementsAttr input, mlir::ElementsAttr& output) const final;
};

mlir::LogicalResult IEDecodeAttributesHooks::decode(mlir::OpaqueElementsAttr input, mlir::ElementsAttr& output) const {
    if (input.getDialect() != IE::IEDialect::getDialectNamespace()) {
        return mlir::failure();
    }

    const auto type = input.getType();
    const auto bytes = input.getValue();

    if (!type.hasStaticShape()) {
        return mlir::failure();
    }
    if (!type.getElementType().isa<mlir::FloatType>() && !type.getElementType().isa<mlir::IntegerType>()) {
        return mlir::failure();
    }

    const auto rawBuffer = makeArrayRef(bytes.data(), bytes.size());

    bool isSplatBuffer = false;
    if (!mlir::DenseElementsAttr::isValidRawBuffer(type, rawBuffer, isSplatBuffer)) {
        return mlir::failure();
    }

    output = mlir::DenseElementsAttr::getFromRawBuffer(type, rawBuffer, isSplatBuffer);
    return mlir::success();
}

}  // namespace

//
// initialize
//

void vpux::IE::IEDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
#undef GET_OP_LIST
            >();

    addInterfaces<IEAsmHooks, IEDecodeAttributesHooks>();
}

//
// materializeConstant
//

mlir::Operation* vpux::IE::IEDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                          mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize IE Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        (void)errorAt(loc, "Can't materialize IE Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
#undef GET_OP_CLASSES
