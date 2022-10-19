//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/m2i_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

mlir::FailureOr<M2iColorFmt> getM2iPlanarOutFmt(mlir::Type oType, IE::ColorFmt cscOutFmt) {
    if (oType.isF16()) {
        if (cscOutFmt == IE::ColorFmt::RGB) {
            return M2iColorFmt::PL_FP16_RGB;
        } else if (cscOutFmt == IE::ColorFmt::BGR) {
            return M2iColorFmt::PL_FP16_BGR;
        }
    } else if (oType.isUnsignedInteger(8)) {
        if (cscOutFmt == IE::ColorFmt::RGB) {
            return M2iColorFmt::PL_RGB24;
        } else if (cscOutFmt == IE::ColorFmt::BGR) {
            return M2iColorFmt::PL_BGR24;
        }
    }
    return mlir::failure();
}

// Check Permute transforms Interleaved->Planar: NHWC(d0,d1,d2,d3) -> NCHW(d0,d3,d1,d2)
inline bool checkPerm(mlir::AffineMap memPerm, mlir::MLIRContext* ctx) {
    const SmallVector<uint32_t> order{0, 3, 1, 2};
    const auto map = mlir::AffineMap::getPermutationMap(makeArrayRef(order), ctx);
    return memPerm == map;
}

// ======================================================================================
// FuseM2iCscResizePl (Planar output)
//   NV12/I420(u8) -> [CSC -> ConvertU8toF16 -> Resize -> Permute] -> Planar_FP16_RGB/BGR
//   NV12/I420(u8) -> [CSC -------------------> Resize -> Permute] -> Planar_U8_RGB/BGR
//

class FuseM2iCscResizePl final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    FuseM2iCscResizePl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MemPermuteOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iCscResizePl::matchAndRewrite(VPU::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!checkPerm(origOp.mem_perm(), getContext())) {
        return mlir::failure();
    }

    mlir::Value sclInput;
    mlir::ArrayAttr sclSizes;
    mlir::ArrayAttr sclAxes;

    if (auto m2iScl = origOp.input().getDefiningOp<VPU::M2IResizeOp>()) {
        sclInput = m2iScl.input();
        sclSizes = m2iScl.sizes();
        sclAxes = m2iScl.axes();
    } else if (auto interp = origOp.input().getDefiningOp<VPU::InterpolateOp>()) {
        sclInput = interp.input();
        sclSizes = interp.sizes_attr().getValue();
        sclAxes = interp.axes_attr().getValue();
    } else {
        return mlir::failure();
    }

    // optional VPU::ConvertOp
    VPU::M2IColorConvertOp m2iCsc;
    if (auto opConvert = sclInput.getDefiningOp<VPU::ConvertOp>()) {
        if (!opConvert.output().getType().cast<vpux::NDTypeInterface>().getElementType().isF16()) {
            return mlir::failure();
        }
        m2iCsc = opConvert.input().getDefiningOp<VPU::M2IColorConvertOp>();
    } else {
        m2iCsc = sclInput.getDefiningOp<VPU::M2IColorConvertOp>();
    }

    if (m2iCsc == nullptr) {
        return mlir::failure();
    }

    const auto iFmt = IEtoM2iColorFmt(m2iCsc.inFmtAttr().getValue());
    const auto oType = origOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto cscOutFmt = m2iCsc.outFmtAttr().getValue();

    const auto res = getM2iPlanarOutFmt(oType, cscOutFmt);
    if (mlir::failed(res)) {
        return mlir::failure();
    }
    const auto oFmt = res.getValue();

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), m2iCsc.input(), true, false, iFmt,
                                                 oFmt, sclSizes, sclAxes, nullptr);
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

// ======================================================================================
// FuseM2iTask2Pl (Planar output)
//     [M2ITaskOp(u8_IL) -> Permute] becomes [M2ITaskOp(u8_PL)]
// Example:
//      NV12/I420(u8) -> [CSC ----> Resize ---> Permute]--> Planar_U8_RGB/BGR
//      NV12/I420(u8) -> [M2ITaskOp(u8_IL) ---> Permute]--> Planar_U8_RGB/BGR
//      NV12/I420(u8) -> [M2ITaskOp(u8_PL)]---------------> Planar_U8_RGB/BGR

class FuseM2iTask2Pl final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    FuseM2iTask2Pl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MemPermuteOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iTask2Pl::matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!checkPerm(origOp.mem_perm(), getContext())) {
        return mlir::failure();
    }

    auto task = origOp.input().getDefiningOp<VPU::M2ITaskOp>();
    if (task == nullptr) {
        return mlir::failure();
    }

    const auto iFmt = task.inFmtAttr().getValue();
    const auto taskOutFmt = task.outFmtAttr().getValue();

    M2iColorFmt oFmt;
    if (taskOutFmt == M2iColorFmt::IL_RGB888) {
        oFmt = M2iColorFmt::PL_RGB24;
    } else if (taskOutFmt == M2iColorFmt::IL_BGR888) {
        oFmt = M2iColorFmt::PL_BGR24;
    } else {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), task.input(), task.do_csc(),
                                                 task.do_norm(), iFmt, oFmt, task.sizesAttr(), task.axesAttr(),
                                                 task.normAttr());
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

// ======================================================================================
// FuseM2iCscResizeIl (Interleaved output)
//   NV12/I420(u8) -> [CSC -> Resize] -> Interleaved_U8_RGB/BGR
//

class FuseM2iCscResizeIl final : public mlir::OpRewritePattern<VPU::M2IResizeOp> {
public:
    FuseM2iCscResizeIl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2IResizeOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2IResizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iCscResizeIl::matchAndRewrite(VPU::M2IResizeOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto m2iScl = origOp;
    auto m2iCsc = origOp.input().getDefiningOp<VPU::M2IColorConvertOp>();
    if (m2iCsc == nullptr) {
        return mlir::failure();
    }

    const auto iFmt = IEtoM2iColorFmt(m2iCsc.inFmtAttr().getValue());
    const auto oFmt = IEtoM2iColorFmt(m2iCsc.outFmtAttr().getValue());

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), m2iCsc.input(), true, false, iFmt,
                                                 oFmt, m2iScl.sizes(), m2iScl.axes(), nullptr);
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

// ======================================================================================
// FuseM2iCscPl (Planar output)
//   NV12/I420(u8) -> [CSC -> {convertU8F16} -> Permute] -> Planar_U8/F16_RGB/BGR
//

class FuseM2iCscPl final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    FuseM2iCscPl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MemPermuteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iCscPl::matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!checkPerm(origOp.mem_perm(), getContext())) {
        return mlir::failure();
    }

    auto opConvert = origOp.input().getDefiningOp<VPU::ConvertOp>();  // optional

    VPU::M2IColorConvertOp m2iCsc;
    if (opConvert == nullptr) {
        m2iCsc = origOp.input().getDefiningOp<VPU::M2IColorConvertOp>();
    } else {
        if (!opConvert.output().getType().cast<vpux::NDTypeInterface>().getElementType().isF16()) {
            return mlir::failure();
        }
        m2iCsc = opConvert.input().getDefiningOp<VPU::M2IColorConvertOp>();
    }

    if (m2iCsc == nullptr) {
        return mlir::failure();
    }

    const auto iFmt = IEtoM2iColorFmt(m2iCsc.inFmtAttr().getValue());
    const auto oType = origOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto cscOutFmt = m2iCsc.outFmtAttr().getValue();

    M2iColorFmt oFmt;
    auto res = getM2iPlanarOutFmt(oType, cscOutFmt);
    if (failed(res)) {
        return mlir::failure();
    } else {
        oFmt = res.getValue();
    }

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), m2iCsc.input(), true, false, iFmt,
                                                 oFmt, nullptr, nullptr, nullptr);
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// FuseM2IOpsPass
//

class FuseM2IOpsPass final : public FuseM2IOpsBase<FuseM2IOpsPass> {
public:
    explicit FuseM2IOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FuseM2IOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX40XX) {
        _log.trace("FuseM2IOpsPass enabled only for VPUX4000 device. Got: {0}", arch);
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FuseM2iCscResizePl>(&ctx, _log);
    patterns.insert<FuseM2iCscResizeIl>(&ctx, _log);
    patterns.insert<FuseM2iCscPl>(&ctx, _log);
    patterns.insert<FuseM2iTask2Pl>(&ctx, _log);

    mlir::ConversionTarget target(ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseM2IOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseM2IOpsPass(Logger log) {
    return std::make_unique<FuseM2IOpsPass>(log);
}
