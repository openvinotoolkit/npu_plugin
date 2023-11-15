//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

VPUIP::StorageElementTableOp findSETableOp(mlir::Value value) {
    auto parentOp = value.getDefiningOp();
    return llvm::TypeSwitch<mlir::Operation*, VPUIP::StorageElementTableOp>(parentOp)
            .Case<VPUIP::StorageElementTableOp>([](VPUIP::StorageElementTableOp seTableOp) {
                return seTableOp;
            })
            .Case<VPUIP::ConcatViewOp>([&](VPUIP::ConcatViewOp) -> VPUIP::StorageElementTableOp {
                VPUX_THROW("Concatenated storage element table operations are not supported");
            })
            .Case<VPUIP::GroupSparseBufferOp>([](VPUIP::GroupSparseBufferOp groupOp) {
                VPUX_THROW_UNLESS(groupOp->getNumOperands() == 3,
                                  "Expected three operands for grouping operation at '{0}', got '{1}'",
                                  groupOp->getLoc(), groupOp->getNumOperands());
                return findSETableOp(groupOp->getOperand(2));
            })
            .Case<VPUIP::NCEClusterTilingOp>([](VPUIP::NCEClusterTilingOp nceClusterTilingOp) {
                auto taskOp = nceClusterTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
                VPUX_THROW_UNLESS(taskOp != nullptr, "Unexpected NCE parent operation at '{0}'",
                                  nceClusterTilingOp->getLoc());
                return findSETableOp(nceClusterTilingOp->getOperand(0));
            })
            .Case<VPUIP::CopyOp>([](VPUIP::CopyOp copyOp) {
                return findSETableOp(copyOp.input());
            })
            .Case<mlir::ViewLikeOpInterface>([](mlir::ViewLikeOpInterface viewOp) {
                return findSETableOp(viewOp.getViewSource());
            })
            .Case<MultiViewOpInterface>([&](vpux::MultiViewOpInterface viewOp) {
                if (auto nceClusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(parentOp)) {
                    auto taskOp = nceClusterOp.getInnerTaskOp();
                    VPUX_THROW_UNLESS(mlir::isa<VPUIP::CopyOp>(taskOp), "Expected copy operation, got '{0}' at '{1}'",
                                      taskOp->getName(), taskOp->getLoc());
                }
                auto opResult = value.dyn_cast<mlir::OpResult>();
                VPUX_THROW_WHEN(opResult == nullptr, "Value '{0}' cannot be converted to an op result", value);
                const auto source = viewOp.getViewSource(opResult.getResultNumber());
                return findSETableOp(source);
            })
            .Default([](mlir::Operation* op) -> VPUIP::StorageElementTableOp {
                VPUX_THROW("Unexpected operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

void computeBasePtrs(VPUIP::StorageElementTableOp seTableOp, VPUIP::DistributedBufferType inputDistType,
                     const Logger& log) {
    log.trace("Computing base pointers for '{0}' at '{1}'", seTableOp->getName(), seTableOp->getLoc());

    if (seTableOp.basePtrs().has_value()) {
        log.nest().trace("Operation already has base pointers computed. Skipping");
        return;
    }

    const auto seDataShape = parseIntArrayAttr<int64_t>(seTableOp.dataShape());
    const auto inputShape = to_small_vector(inputDistType.getShape());
    VPUX_THROW_UNLESS(seDataShape == inputShape,
                      "Mismatch between storage element table's data shape '{0}' and input shape '{1}'", seDataShape,
                      inputShape);

    auto perClusterOffsets = inputDistType.getPerClusterMemoryShapeOffsets();
    auto perClusterShapes = inputDistType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterOffsets.size() == perClusterShapes.size(),
                      "Mismatch between per cluster offsets '{0}' and shapes '{1}", perClusterOffsets.size(),
                      perClusterShapes.size());

    const auto findCluster = [&](const int64_t h, const int64_t w) -> int32_t {
        for (auto p : zip(perClusterOffsets, perClusterShapes) | indexed) {
            const auto offsets = std::get<0>(p.value());
            const auto shape = std::get<1>(p.value());
            auto containsH = offsets[Dims4D::Act::H] <= h && h < (offsets[Dims4D::Act::H] + shape[Dims4D::Act::H]);
            auto containsW = offsets[Dims4D::Act::W] <= w && w < (offsets[Dims4D::Act::W] + shape[Dims4D::Act::W]);
            if (containsH && containsW) {
                return checked_cast<int32_t>(p.index());
            }
        }
        return static_cast<int32_t>(-1);
    };

    const auto outputNDType = seTableOp.getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputNDType.getShape();
    const auto outputH = outputShape[Dims4D::Act::H];
    const auto outputW = outputShape[Dims4D::Act::W];

    const auto seAttr = seTableOp.seAttr().value_or(nullptr);
    const auto seSize = seTableOp.seSize();
    const auto seDepth = seTableOp.seDepth();

    const auto numElements = outputNDType.getNumElements();
    std::vector<int32_t> basePtrs(numElements, 0);

    for (int64_t h = 0; h < outputH; ++h) {
        for (int64_t w = 0; w < outputW; ++w) {
            for (int64_t se = 0; se < seDepth; ++se) {
                const Shape outputOffsets{0, se * seSize, h, w};
                auto inputOffsets =
                        (seAttr != nullptr) ? seAttr.backInferCoord(outputOffsets, Shape(inputShape)) : outputOffsets;
                const auto seSpatialOffset = (h * outputW + w) * seDepth;
                basePtrs[seSpatialOffset + se] =
                        findCluster(inputOffsets[Dims4D::Act::H], inputOffsets[Dims4D::Act::W]);
            }
        }
    }

    const auto basePtrType =
            mlir::RankedTensorType::get({static_cast<int64_t>(basePtrs.size())}, getInt32Type(seTableOp.getContext()));
    const auto basePtrsElems = mlir::DenseIntElementsAttr::get(basePtrType, basePtrs);
    seTableOp.basePtrsAttr(basePtrsElems);
}

//
// ComputeSEBasePtrsPass
//

class ComputeSEBasePtrsPass final : public VPUIP::ComputeSEBasePtrsBase<ComputeSEBasePtrsPass> {
public:
    explicit ComputeSEBasePtrsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ComputeSEBasePtrsPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.input_storage_element_table() == nullptr) {
            return;
        }

        VPUX_THROW_WHEN(nceOp.task_type() == VPUIP::NCETaskType::ELTWISE,
                        "Eltwise operations with input storage element tables are not yet supported");

        auto parentTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        if (parentTilingOp == nullptr) {
            return;
        }

        auto inputArg = nceOp.input().dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(inputArg == nullptr, "Input is not a block argument");
        auto outerInput = parentTilingOp.getOperand(inputArg.getArgNumber());

        if (!outerInput.getType().isa<VPUIP::DistributedBufferType>()) {
            return;
        }

        _log.trace("Found NCE operation at '{0}'", nceOp->getLoc());

        auto inputDistType = outerInput.getType().cast<VPUIP::DistributedBufferType>();

        auto seTableBlockArg = nceOp.input_storage_element_table().dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(seTableBlockArg == nullptr, "Input storage element table is not a block argument");
        auto outerSeTable = parentTilingOp.getOperand(seTableBlockArg.getArgNumber());
        auto seTableOp = findSETableOp(outerSeTable);

        computeBasePtrs(seTableOp, inputDistType, _log);
    });
}

}  // namespace

//
// createComputeSEBasePtrsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createComputeSEBasePtrsPass(Logger log) {
    return std::make_unique<ComputeSEBasePtrsPass>(log);
}
