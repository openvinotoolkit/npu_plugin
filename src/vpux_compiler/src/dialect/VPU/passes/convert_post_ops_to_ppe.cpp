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

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/ppe_utils.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// ConvertPostOpsToPPE
//

class ConvertPostOpsToPPEPass final : public ConvertPostOpsToPPEBase<ConvertPostOpsToPPEPass> {
public:
    explicit ConvertPostOpsToPPEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    template <class ConcreteOp>
    void updatePPETasks(ConcreteOp op, VPU::ArchKind arch);

    void updateNCEEltwisePPETasks(VPU::NCEEltwiseOp op, VPU::ArchKind arch);

private:
    Logger _log;
};

template <class ConcreteOp>
void ConvertPostOpsToPPEPass::updatePPETasks(ConcreteOp op, VPU::ArchKind arch) {
    if (!op.post_op().hasValue()) {
        return;
    }

    const auto inElemType = op.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = op.output().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto postOpParams = parsePostOp(op.post_opAttr(), inElemType, outElemType, arch, op.getLoc());

    if (!postOpParams.hasValue()) {
        return;
    }

    const auto ppeTask = getPPEAttr(postOpParams.getValue(), op.getContext());
    op.ppeAttr(ppeTask);
    op.removePost_opAttr();
}

void ConvertPostOpsToPPEPass::updateNCEEltwisePPETasks(VPU::NCEEltwiseOp op, VPU::ArchKind arch) {
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;

    auto origOutType = op.output().getType().cast<vpux::NDTypeInterface>();
    auto outElemType = origOutType.getElementType();
    const auto inElemType = op.input1().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (auto outElemQType = outElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outElemType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    auto* ctx = op.getContext();

    auto postOp = op.post_op().hasValue() ? op.post_opAttr() : nullptr;
    const auto postOpParams = parsePostOp(postOp, inElemType, outElemType, arch, op.getLoc());
    if (postOpParams.hasValue()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
    }

    VPU::PPEMode ppeType = VPU::getPPEMode(op.op_type());
    auto ppeAttr = getPPETaskAttr(ctx, ppeType);

    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto quantScale = VPU::calculateQuantScaleVectorForEltwise(
            op.input1().getType().cast<mlir::ShapedType>(), op.input2().getType().cast<mlir::ShapedType>(),
            op.output().getType().cast<mlir::ShapedType>(), arch, op.op_type() == VPU::EltwiseType::MULTIPLY);
    if (quantScale.hasValue()) {
        const auto scale = quantScale.getValue();

        const auto mult = getQuantMultFromScale(scale);
        const auto shifts = getQuantShiftAndPostShiftFromScale(scale);

        const auto shift = shifts.first;
        const auto post_shift = shifts.second;

        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift, ArrayRef<int64_t>{mult},
                                 ArrayRef<int64_t>{shift}, post_shift);
    } else {
        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift);
    }

    op.ppeAttr(ppeAttr);
    // Can't have both 'post_op' and 'ppe' attributes at the same time
    op.removePost_opAttr();
}

void ConvertPostOpsToPPEPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const auto callback = [&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                    updatePPETasks<VPU::NCEConvolutionOp>(op, arch);
                })
                .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp op) {
                    updatePPETasks<VPU::NCEMaxPoolOp>(op, arch);
                })
                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                    updatePPETasks<VPU::NCEDepthConvolutionOp>(op, arch);
                })
                .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                    updateNCEEltwisePPETasks(op, arch);
                });
    };

    func.walk(callback);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertPostOpsToPPEPass(Logger log) {
    return std::make_unique<ConvertPostOpsToPPEPass>(log);
}
