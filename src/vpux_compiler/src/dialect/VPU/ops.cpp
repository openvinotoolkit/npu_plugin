//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult VPU::sameLayout(VPU::DistributedTensorType inDistributedType,
                                    VPU::DistributedTensorType outDistributedType, LogCb logCb) {
    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getOrder(),
                      outDistributedType.getOrder()));
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult VPU::sameLayout(VPUIP::DistributedBufferType inDistributedType,
                                    VPUIP::DistributedBufferType outDistributedType, LogCb logCb) {
    auto isContinuousWithSameOrder = [&]() {
        const auto inStrideReqs = StrideReqs::compact(inDistributedType.getShape().size());
        const auto outStrideReqs = StrideReqs::compact(outDistributedType.getShape().size());
        auto inRes = inStrideReqs.checkStrides(inDistributedType);
        auto outRes = outStrideReqs.checkStrides(outDistributedType);
        return inRes && outRes && inDistributedType.getDimsOrder() == outDistributedType.getDimsOrder();
    };

    // The strides will be checked when comparing the layouts. So the function will return true if the layouts are equal
    // or the buffers are compact with same dim order
    if (inDistributedType.getLayout() != outDistributedType.getLayout() && !isContinuousWithSameOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getLayout(),
                      outDistributedType.getLayout()));
        return mlir::failure();
    }
    return mlir::success();
}

bool VPU::isVFNCESupported(VPU::NCEOpInterface op) {
    auto isOne = [](auto val) {
        return val == 1;
    };

    if (llvm::all_of(op.getStridesVal(), isOne)) {
        return true;
    }

    return false;
}

//
// materializeConstant
//

mlir::Operation* vpux::VPU::VPUDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                            mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops.cpp.inc>

namespace {

//
// LayerWithPermuteInterface
//

template <class MainOpType>
class LayerWithPermuteInterface final :
        public IE::LayerWithPermuteInterface::ExternalModel<LayerWithPermuteInterface<MainOpType>, MainOpType> {
public:
    bool isSupportedPermutation(mlir::Operation* nceOp, mlir::Operation* permuteOp) const {
        if (VPU::getCompilationMode(permuteOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!isSupportedODUPermute(permuteOp)) {
            return false;
        }

        const auto outputShape = getShape(nceOp->getResult(0));
        const auto outputBatch = outputShape[Dims4D::Act::N];
        if (outputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
            return false;
        }

        return VPUIP::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(nceOp)).succeeded();
    }

private:
    bool isSupportedODUPermute(mlir::Operation* permuteOp) const {
        if (!mlir::isa<IE::ReorderOp, IE::MemPermuteOp>(permuteOp)) {
            return false;
        }

        const auto module = permuteOp->getParentOfType<mlir::ModuleOp>();
        const auto arch = VPU::getArch(module);

        // TODO remove this check. Attach the interface according to architecture.
        // E#84642
        const mlir::DenseSet<VPU::ArchKind> supportedTargets = {
                VPU::ArchKind::VPUX37XX,
        };

        if (!supportedTargets.contains(arch)) {
            return false;
        }

        // Check that reorder is not applied to sub-byte element types:
        const auto elemType = permuteOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const Bit elemSize = vpux::getElemTypeSize(elemType);
        if (elemSize.count() < CHAR_BIT) {
            return false;
        }

        // Check that permutation is supported by ODU
        const auto outOrder = DimsOrder::fromValue(permuteOp->getResult(0));
        const std::unordered_set<DimsOrder> supportedOrders = {
                DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHCW, DimsOrder::NHWC, DimsOrder::NWCH, DimsOrder::NWHC,
        };

        if (supportedOrders.count(outOrder) != 1) {
            return false;
        }

        auto maybeMemPermute = mlir::dyn_cast_or_null<IE::MemPermuteOp>(permuteOp);
        if (maybeMemPermute == nullptr) {
            // IE.Reorder does not need any additional checks.
            return true;
        }

        // IE.MemPermute must produce such target orders that they are compatible with ODU.
        const auto inOrder = DimsOrder::fromValue(maybeMemPermute.input());
        const auto memPerm = maybeMemPermute.mem_perm();
        const auto targetOrder = vpux::applyPermutation(inOrder, DimsOrder::fromAffineMap(memPerm));

        return supportedOrders.count(targetOrder) == 1;
    }
};

//
// redirectLayerWithPermuteInterfaceForIE
//

void redirectLayerWithPermuteInterfaceForIE(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::GroupConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<LayerWithPermuteInterface<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<LayerWithPermuteInterface<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<LayerWithPermuteInterface<IE::AddOp>>(*ctx);
    });
}

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::VPUDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    redirectLayerWithPermuteInterfaceForIE(registry);
}
