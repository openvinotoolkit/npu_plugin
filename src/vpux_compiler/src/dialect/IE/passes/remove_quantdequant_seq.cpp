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
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

using namespace vpux;

namespace {

//
// RemoveQuantDequantSeqPass
//

class RemoveQuantDequantSeqPass final : public IE::RemoveQuantDequantSeqBase<RemoveQuantDequantSeqPass> {
public:
    explicit RemoveQuantDequantSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    // TODO: Code base on fuse_post_ops pass, how to make it common
    bool canBeMergedWithPreceedingDpu(mlir::Operation* op, IE::ClampOp clampOp) {
        if (!op->hasOneUse()) {
            return false;
        }
        auto producerOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (producerOp == nullptr) {
            return false;
        }
        if (!producerOp.isSupportedPostOp(clampOp)) {
            return false;
        }
        if (producerOp.getPostOp().hasValue()) {
            return false;
        }
        return true;
    }
};

void RemoveQuantDequantSeqPass::safeRunOnFunc() {
    auto func = getFunction();
    // Remove remaining Quantize->Dequantize sequence to not perform explicit FakeQuantize.
    // This might have slight impact on accuracy but gives visible performance improvement
    // TODO: Evaluate possibility of replacing such sequence with ClampOp fused with DPU task
    uint32_t convFused = 0;
    uint32_t grConvFused = 0;
    uint32_t maxPoolFused = 0;
    func.walk([this, &grConvFused, &convFused, &maxPoolFused](vpux::IE::QuantizeOp quantizeOp) {
        if (!quantizeOp->hasOneUse()) {
            return;
        }
        auto dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*quantizeOp->getUsers().begin());
        if (dequantizeOp == nullptr) {
            return;
        }

        const auto elemType = quantizeOp.getType().cast<mlir::ShapedType>().getElementType();
        const auto quantType = elemType.dyn_cast<mlir::quant::QuantizedType>();
        VPUX_THROW_UNLESS(quantType != nullptr, "Type must be quantized, but provided {0}", elemType);
        int64_t levels;
        float rMin, rMax;
        if (const auto uniformType = quantType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            getFakeQuantParams(uniformType, levels, rMin, rMax);
            mlir::OpBuilder builder(dequantizeOp);
            const auto min = getFPAttr(&getContext(), rMin);
            const auto max = getFPAttr(&getContext(), rMax);
            const auto broadcastType =
                    vpux::IE::AutoBroadcastTypeAttr::get(&getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
            auto andOp = builder.create<IE::AndOp>(quantizeOp->getLoc(), quantizeOp->getOperand(0),
                                                   quantizeOp->getOperand(0), broadcastType, nullptr);
            auto clampOp = builder.create<IE::ClampOp>(quantizeOp->getLoc(), quantizeOp->getOperand(0), min, max);
            if (!canBeMergedWithPreceedingDpu(quantizeOp.input().getDefiningOp(), clampOp)) {
                clampOp->setOperand(0, andOp);
            }
            dequantizeOp.replaceAllUsesWith(clampOp.getOperation());
        }
    });
}

}  // namespace

//
// createRemoveQuantDequantSeqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveQuantDequantSeqPass(Logger log) {
    return std::make_unique<RemoveQuantDequantSeqPass>(log);
}
