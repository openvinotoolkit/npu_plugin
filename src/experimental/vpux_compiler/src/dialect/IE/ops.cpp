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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"

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
    if (const auto affineMapAttr = attr.dyn_cast<mlir::AffineMapAttr>()) {
        if (const auto dimsOrder = DimsOrder::fromAffineMap(affineMapAttr.getValue())) {
            if (const auto name = dimsOrder->getCanonicalName()) {
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
    if (input.getDialect()->getTypeID() != mlir::TypeID::get<IE::IEDialect>()) {
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
    if (!value.isa<ConstContentAttr>()) {
        errorAt(loc, "Can't materialize IE Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        errorAt(loc, "Can't materialize IE Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<IE::ConstantOp>(loc, type, value.cast<mlir::ElementsAttr>());
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
#undef GET_OP_CLASSES
