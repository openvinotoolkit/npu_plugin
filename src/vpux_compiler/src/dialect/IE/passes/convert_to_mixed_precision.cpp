//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {
template <class ConcreteOp>
bool isMixPrecisionSupported(ConcreteOp op, const VPU::ArchKind& arch) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MaxPoolOp>(op.getOperation())) {
        return false;
    }

    if (arch == VPU::ArchKind::VPUX37XX) {
        // HW limitations below do not apply to VPUX37XX
        // However, leaky ReLU does not work accurately in mixed mode.
        // Covered in E#34900
        const auto hasLeakyReLU = llvm::any_of(op->getUsers(), [](mlir::Operation* op) {
            return mlir::isa<IE::LeakyReluOp>(op);
        });
        // MaxPool has accuracy issues in mixed mode.
        // Covered in E#50414
        const auto isMaxPool = mlir::isa<IE::MaxPoolOp>(op.getOperation());

        // Thus, mixed precision is supported when consumer is not leaky ReLU and the operation is not max pooling
        return !hasLeakyReLU && !isMaxPool;
    }

    // NOTE: HW limitation, in mixed mode the grids of the MPEs are conflicting between
    // each other, which leads to 1x1 workloads.
    auto outputShape = getShape(op.output());
    return outputShape[Dims4D::Act::H] == 1 && outputShape[Dims4D::Act::W] == 1;
}

bool areAnyUserQuantizeOps(mlir::Operation* op) {
    return llvm::any_of(op->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::QuantizeOp>(op);
    });
}

/*
 *  Bias will be rescaled for mixed precision and written in weight table later, so need to check whether the
 *  rescaled bias range exceeds or not
 */
template <class ConcreteOp>
mlir::LogicalResult checkRescaledBiasRange(ConcreteOp op) {
    auto inputDequantizeOp = op.input().template getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = op.filter().template getDefiningOp<IE::DequantizeOp>();
    if (!inputDequantizeOp || !filterDequantizeOp) {
        return mlir::failure();
    }

    if (auto biasAttr = op.bias()) {
        const auto inElemType =
                inputDequantizeOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
        const auto filterElemType =
                filterDequantizeOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
        auto biasConstOp = biasAttr.template getDefiningOp<Const::DeclareOp>();
        const auto bias = biasConstOp.contentAttr();
        const auto OC = getShape(op.filter())[Dims4D::Filter::OC];
        if (mlir::failed(VPU::NCESparsity::getRescaledBias(bias, inElemType, filterElemType, OC))) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

//
// ConvertToMixedPrecisionPass
//

class ConvertToMixedPrecisionPass final : public IE::ConvertToMixedPrecisionBase<ConvertToMixedPrecisionPass> {
public:
    explicit ConvertToMixedPrecisionPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

class MixedPrecisionConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    MixedPrecisionConvRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convolutionOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult MixedPrecisionConvRewriter::matchAndRewrite(IE::ConvolutionOp convolutionOp,
                                                                mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(convolutionOp) || !isMixPrecisionSupported(convolutionOp, _arch)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(convolutionOp))) {
        return mlir::failure();
    }
    auto dequantizeOp = convolutionOp.input().getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = convolutionOp.filter().getDefiningOp<IE::DequantizeOp>();

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            convolutionOp, convolutionOp.getType(), dequantizeOp.input(), filterDequantizeOp.input(),
            convolutionOp.bias(), convolutionOp.strides(), convolutionOp.pads_begin(), convolutionOp.pads_end(),
            convolutionOp.dilations(), convolutionOp.post_opAttr());

    return mlir::success();
}

class MixedPrecisionGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    MixedPrecisionGroupConvRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult MixedPrecisionGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(groupConvolutionOp) || !isMixPrecisionSupported(groupConvolutionOp, _arch)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(groupConvolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeOp = groupConvolutionOp.input().getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = groupConvolutionOp.filter().getDefiningOp<IE::DequantizeOp>();

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            groupConvolutionOp, groupConvolutionOp.getType(), dequantizeOp.input(), filterDequantizeOp.input(),
            groupConvolutionOp.bias(), groupConvolutionOp.strides(), groupConvolutionOp.pads_begin(),
            groupConvolutionOp.pads_end(), groupConvolutionOp.dilations(), groupConvolutionOp.groupsAttr(),
            groupConvolutionOp.post_opAttr());

    return mlir::success();
}

class MixedPrecisionMaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MixedPrecisionMaxPoolRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp maxPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult MixedPrecisionMaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp maxPoolOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(maxPoolOp) || !isMixPrecisionSupported(maxPoolOp, _arch)) {
        return mlir::failure();
    }
    auto dequantizeOp = maxPoolOp.input().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            maxPoolOp, maxPoolOp.getType(), dequantizeOp.input(), maxPoolOp.kernel_size(), maxPoolOp.strides(),
            maxPoolOp.pads_begin(), maxPoolOp.pads_end(), maxPoolOp.rounding_type(), maxPoolOp.post_opAttr());

    return mlir::success();
}

void ConvertToMixedPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MixedPrecisionConvRewriter>(&ctx, arch, _log);
    patterns.insert<MixedPrecisionGroupConvRewriter>(&ctx, arch, _log);
    patterns.insert<MixedPrecisionMaxPoolRewriter>(&ctx, arch, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMixedPrecision
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToMixedPrecision(Logger log) {
    return std::make_unique<ConvertToMixedPrecisionPass>(log);
}
