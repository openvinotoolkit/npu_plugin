//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TensorEncoding.h>
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
    AliasResult getAlias(mlir::Attribute attr, llvm::raw_ostream& os) const final;
    AliasResult getAlias(mlir::Type type, llvm::raw_ostream& os) const final;
};

IEAsmHooks::AliasResult IEAsmHooks::getAlias(mlir::Attribute attr, llvm::raw_ostream& os) const {
    if (const auto mapAttr = attr.dyn_cast<mlir::AffineMapAttr>()) {
        const auto map = mapAttr.getValue();

        if (map.isPermutation()) {
            const auto dimsOrder = DimsOrder::fromAffineMap(map);

            if (const auto name = dimsOrder.getCanonicalName()) {
                os << name.getValue();
                return AliasResult::FinalAlias;
            }
        }
    }

    return AliasResult::NoAlias;
}

IEAsmHooks::AliasResult IEAsmHooks::getAlias(mlir::Type type, llvm::raw_ostream& os) const {
    if (type.isa<mlir::quant::QuantizedType>()) {
        os << "qElemType";
        return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
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

//
// TensorEncodingVerifier
//

class TensorEncodingVerifier final :
        public mlir::VerifiableTensorEncoding::ExternalModel<TensorEncodingVerifier, IE::TensorAttr> {
public:
    mlir::LogicalResult verifyEncoding(mlir::Attribute attr, ArrayRef<int64_t> shape, mlir::Type,
                                       FuncRef<mlir::InFlightDiagnostic()> emitError) const {
        const auto desc = attr.dyn_cast<IE::TensorAttr>();

        if (desc == nullptr) {
            return printTo(emitError(), "Unsupported TensorType encoding '{0}'", attr);
        }

        if (const auto orderAttr = desc.order()) {
            const auto map = orderAttr.getValue();

            if (!map.isPermutation()) {
                return printTo(emitError(), "TensorType order '{0}' is not a permutation", map);
            }

            if (checked_cast<size_t>(map.getNumResults()) != shape.size()) {
                return printTo(emitError(), "TensorType order '{0}' doesn't match to shape '{1}'", map, shape);
            }
        }

        return mlir::success();
    }
};

}  // namespace

//
// initialize
//

void vpux::IE::IEDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
            >();

    addInterfaces<IEAsmHooks, IEDecodeAttributesHooks>();

    IE::TensorAttr::attachInterface<TensorEncodingVerifier>(*getContext());
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

bool IE::isActShaveKernel(mlir::Operation* operation) {
    const auto module = operation->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    if (arch != VPU::ArchKind::VPUX37XX) {
        return false;
    }

    return VPUIP::NCEInvariant::verifyKernel(operation, Logger::global()).failed() &&
           operation->hasTrait<IE::EltwiseOp>();
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
