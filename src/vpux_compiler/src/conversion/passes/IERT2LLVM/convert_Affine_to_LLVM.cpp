//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

// TODO: E66812, it should be sufficient to have warnings disabled for 3-rd parties
// in CMake but it does not work for early versions of MSVC 2019
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>

using namespace vpux;

namespace {

class ConvertAffine2LLVMPass final : public ConvertAffine2LLVMBase<ConvertAffine2LLVMPass> {
public:
    explicit ConvertAffine2LLVMPass(Logger log) {
        VPUX_UNUSED(log);
    }

private:
    void safeRunOnModule() final;
};

// The RewritePattern below is taken from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L149
//   (referred from
//   https://discourse.llvm.org/t/support-for-lowering-to-llvm-for-the-standard-dialects-maxfop-and-minfop-operations/63588/3)
template <typename OpTy, mlir::arith::CmpFPredicate pred>
struct MaxMinFOpConverter : public mlir::OpRewritePattern<OpTy> {
public:
    using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(OpTy op, mlir::PatternRewriter& rewriter) const final {
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        mlir::Location loc = op.getLoc();
        // If any operand is NaN, 'cmp' will be true (and 'select' returns 'lhs').
        static_assert(pred == mlir::arith::CmpFPredicate::UGT || pred == mlir::arith::CmpFPredicate::ULT,
                      "pred must be either UGT or ULT");
        mlir::Value cmp = rewriter.create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
        mlir::Value select = rewriter.create<mlir::SelectOp>(loc, cmp, lhs, rhs);

        // Handle the case where rhs is NaN: 'isNaN(rhs) ? rhs : select'.
        mlir::Value isNaN = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNO, rhs, rhs);
        rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, isNaN, rhs, select);
        return mlir::success();
    }
};

void ConvertAffine2LLVMPass::safeRunOnModule() {
    auto& ctx = getContext();
    mlir::LLVMConversionTarget target(ctx);
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&ctx);

    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateLoopToStdConversionPatterns(patterns);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

    patterns.add<MaxMinFOpConverter<mlir::arith::MaxFOp, mlir::arith::CmpFPredicate::UGT>,
                 MaxMinFOpConverter<mlir::arith::MinFOp, mlir::arith::CmpFPredicate::ULT>>(&ctx);

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

}  // namespace

//
// createConvertAffine2LLVMPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertAffine2LLVMPass(Logger log) {
    return std::make_unique<ConvertAffine2LLVMPass>(log);
}
