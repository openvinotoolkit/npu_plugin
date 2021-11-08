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

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>

using namespace vpux;

namespace {

//
// MemRefAttrLayout
//

class MemRefAttrLayout final :
        public mlir::MemRefLayoutAttrInterface::ExternalModel<MemRefAttrLayout, IERT::MemRefAttr> {
public:
    mlir::AffineMap getAffineMap(mlir::Attribute attr) const {
        const auto desc = attr.dyn_cast<IERT::MemRefAttr>();
        VPUX_THROW_WHEN(desc == nullptr, "Unsupported MemRef layout '{0}'", attr);

        const auto orderMap = desc.order().getValue();

        const auto elemStrides = parseIntArrayAttr<int64_t>(desc.strides());
        const auto stridesMap = mlir::makeStridedLinearLayoutMap(elemStrides, 0, attr.getContext());

        return stridesMap.compose(orderMap);
    }

    bool isIdentity(mlir::Attribute) const {
        return false;
    }

    mlir::LogicalResult verifyLayout(mlir::Attribute attr, ArrayRef<int64_t> shape,
                                     FuncRef<mlir::InFlightDiagnostic()> emitError) const {
        const auto desc = attr.dyn_cast<IERT::MemRefAttr>();
        if (desc == nullptr) {
            return printTo(emitError(), "Unsupported MemRef layout '{0}'", attr);
        }

        if (!desc.order().getValue().isPermutation()) {
            return printTo(emitError(), "Dims order '{0}' is not a permutation affine map", desc.order());
        }

        const auto order = DimsOrder::fromAffineMap(desc.order().getValue());
        const auto elemStrides = parseIntArrayAttr<int64_t>(desc.strides());

        const auto memShape = order.toMemoryOrder(ShapeRef(shape));

        const auto elemSize = 1_Bit;
        const auto strides = Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                                         return stride * elemSize;
                                                     })));
        const auto memStrides = order.toMemoryOrder(strides);

        const auto reqs = StrideReqs::simple(shape.size());

        if (!reqs.checkStrides(memStrides, elemSize, memShape)) {
            return printTo(emitError(), "Strides '{0}' do not match with shape '{1}' and order '{2}'", desc.strides(),
                           shape, order);
        }

        return mlir::success();
    }
};

}  // namespace

//
// initialize
//

void vpux::IERT::IERTDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IERT/generated/ops.cpp.inc>
            >();

    IERT::MemRefAttr::attachInterface<MemRefAttrLayout>(*getContext());
}

//
// materializeConstant
//

mlir::Operation* vpux::IERT::IERTDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                              mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize IERT Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize IERT Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, eraseTiledInfo(type.cast<mlir::MemRefType>()),
                                            value.cast<Const::ContentAttr>());
}

//
// Operation executor attributes
//

namespace {

constexpr StringLiteral executorAttrName = "IERT.executor";
constexpr StringLiteral numUnitsAttrName = "IERT.num_units";

}  // namespace

void vpux::IERT::IERTDialect::setExecutor(mlir::async::ExecuteOp execOp, mlir::Attribute executor, uint32_t numUnits) {
    execOp->setAttr(executorAttrName, executor);
    execOp->setAttr(numUnitsAttrName, getIntAttr(execOp->getContext(), numUnits));
}

llvm::StringLiteral vpux::IERT::IERTDialect::getExecutorAttrName() {
    return executorAttrName;
}

mlir::Attribute vpux::IERT::IERTDialect::getExecutor(mlir::async::ExecuteOp execOp, uint32_t& numUnits) {
    if (const auto executor = execOp->getAttr(executorAttrName)) {
        const auto numUnitsAttr = execOp->getAttr(numUnitsAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
        VPUX_THROW_UNLESS(numUnitsAttr != nullptr,
                          "'{0}' attribute was not set, it must be used together with '{1}' attribute", numUnitsAttr,
                          executorAttrName);

        numUnits = checked_cast<uint32_t>(numUnitsAttr.getInt());
        return executor;
    }

    VPUX_THROW("Can't find Executor attributes for Operation at '{0}'", execOp->getLoc());
}

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IERT/generated/ops.cpp.inc>
