//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/utils/core/algo.hpp"
#include "vpux/utils/core/enums.hpp"

using namespace vpux;

namespace {

llvm::unique_function<int32_t(size_t)> getMultShiftFunc(EMU::NCEClusterTaskOp op, VPU::ArchKind arch, size_t OC) {
    SmallVector<int64_t> ppeQuantMult = {1};
    SmallVector<int64_t> ppeQuantShift = {0};

    for (auto ppeOp : op.getPpe().getOps<EMU::PPETaskOp>()) {
        if (ppeOp.getQuantMult().has_value()) {
            ppeQuantMult = parseIntArrayAttr<int64_t>(ppeOp.getQuantMult().value());
        }
        if (ppeOp.getQuantShift().has_value()) {
            ppeQuantShift = parseIntArrayAttr<int64_t>(ppeOp.getQuantShift().value());
        }
    }

    broadcast(ppeQuantMult, OC);
    broadcast(ppeQuantShift, OC);

    const auto ppeConverter = VPU::NCESparsity::ppeConvertersMap.at(arch);

    auto inElemType = op.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    return [ppeConverter, ppeQuantMult = std::move(ppeQuantMult), ppeQuantShift = std::move(ppeQuantShift),
            inElemType](size_t oc) {
        auto multShift = ppeConverter(checked_cast<uint8_t>(ppeQuantShift[oc]),
                                      checked_cast<uint16_t>(ppeQuantMult[oc]), 1, inElemType);
        return multShift;
    };
}

std::vector<int32_t> getWeightsTable(EMU::NCEClusterTaskOp op, VPU::ArchKind arch, int64_t OC) {
    auto getMultShift = getMultShiftFunc(op, arch, checked_cast<size_t>(OC));

    std::vector<std::int32_t> weightsTableVals(OC * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    for (auto oc : irange(checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd] = 0;
        weightsTableVals[wtInd + 1] = 0;
        weightsTableVals[wtInd + 2] = getMultShift(oc);
        weightsTableVals[wtInd + 3] = 0;
    }

    return weightsTableVals;
}

//
// AddWeightsTable
//

class AddWeightsTable final : public mlir::OpRewritePattern<EMU::NCEClusterTaskOp> {
public:
    AddWeightsTable(mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch)
            : mlir::OpRewritePattern<EMU::NCEClusterTaskOp>(ctx), _log(log), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(EMU::NCEClusterTaskOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
};

mlir::LogicalResult AddWeightsTable::matchAndRewrite(EMU::NCEClusterTaskOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    const auto outputShape = getShape(origOp.getOutput());
    const auto OC = outputShape[Dims4D::Act::C];

    SmallVector<int64_t> weightTableShape{OC, 1, 1, VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};
    auto weightsTable = getWeightsTable(origOp, _arch, OC);

    const auto dataType = mlir::RankedTensorType::get(weightTableShape, getSInt32Type(getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataType, makeArrayRef(weightsTable));

    auto weightsTableConstOp =
            rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataType, Const::ContentAttr::get(dataAttr));

    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), origOp.getWeights(), weightsTableConstOp.getOutput(),
            origOp.getTaskType(), origOp.getKernelSizeAttr(), origOp.getKernelStridesAttr(),
            origOp.getKernelPaddingAttr(), origOp.getRawFilterShapeAttr());

    nceOp.addPPETask(rewriter, origOp);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// AddWeightsTableToEmuPass
//

class AddWeightsTableToEmuPass final : public EMU::AddWeightsTableToEltwiseOpsBase<AddWeightsTableToEmuPass> {
public:
    explicit AddWeightsTableToEmuPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnFunc() final;
};

/*
    In board case, quant scale (mult, shift) settings are provided as part of outtput TensorReference for Eltwise or
   AvgPool operations which don't have weights table. Since in emulator we can parse the weights table for them as well,
   in this pass we're constructing a weights table from the PPE quantization settings.
*/
void AddWeightsTableToEmuPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    VPUX_THROW_UNLESS(VPU::NCESparsity::ppeConvertersMap.find(arch) != VPU::NCESparsity::ppeConvertersMap.end(),
                      "Failed to map PPE converter to target arch");

    const auto isLegalOp = [&](EMU::NCEClusterTaskOp op) {
        return (op.getTaskType() != VPUIP::NCETaskType::ELTWISE && op.getTaskType() != VPUIP::NCETaskType::AVEPOOL) ||
               op.getWeightTable() != nullptr;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<EMU::NCEClusterTaskOp>(isLegalOp);
    target.addLegalOp<EMU::PPETaskOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AddWeightsTable>(&ctx, _log, arch);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::EMU::createAddWeightsTableToEmuPass(Logger log) {
    return std::make_unique<AddWeightsTableToEmuPass>(log);
}
