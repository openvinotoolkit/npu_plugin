//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include <vpux/hwtest/ops/act_shave_op.hpp>

#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"

namespace vpux {
namespace hwtest {

namespace {

IERT::KernelInfo getKernelInfo(nb::ActivationLayer activation, mlir::MLIRContext* ctx) {
    switch (activation.activationType) {
    case nb::ActivationType::HSwish:
        return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"hswish_fp16"}, {"hswish_fp16.cpp"}};
    case nb::ActivationType::Sigmoid:
        return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"sigmoid_fp16"}, {"sigmoid_fp16.c"}};
    case nb::ActivationType::Softmax:
        return IERT::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, activation.axis)},
                                {"singleShaveSoftmax"},
                                {"single_shave_softmax.cpp"}};
    default:
        VPUX_THROW("Only HSwish, Sigmoid or Softmax activations is supported for ActShave tests");
    }
}

}  // namespace

void buildActShaveTask(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                       Logger& log, SmallVector<mlir::Type> inputTypes, vpux::VPURT::DeclareBufferOp inputCMX,
                       vpux::VPURT::DeclareBufferOp outputCMX, mlir::ValueRange waitBarrier,
                       mlir::ValueRange updateBarrier, size_t cluster, size_t /*unit*/) {
    auto* ctx = builder.getContext();
    auto activation = testDesc.getActivationLayer();
    auto kernelInfo = getKernelInfo(activation, ctx);

    const auto convertToUnrankedType = [ctx](mlir::Type srcType) -> mlir::Type {
        auto type = srcType.dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), mlir::SymbolRefAttr::get(VPU::MemoryKindAttr::get(
                                                                            ctx, VPU::MemoryKind::CMX_NN)));
    };
    SmallVector<mlir::Type> inputTypesUnranked;
    std::transform(inputTypes.begin(), inputTypes.end(), std::back_inserter(inputTypesUnranked), convertToUnrankedType);
    std::transform(kernelInfo.args.begin(), kernelInfo.args.end(), std::back_inserter(inputTypesUnranked),
                   [](mlir::Attribute arg) {
                       return arg.getType();
                   });

    // first creating management kernel definition
    VPUIP::createRuntimeKernelDefinition(module, log);

    // Create built-in function ------------------------------------------------

    SmallString builtInFunctionName{"builtin_actshave"};

    auto builtInFunction = VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypesUnranked,
                                                        kernelInfo.entryName, kernelInfo.sourceFileName, log);

    // Spawn Task: Kernel ------------------------------------------------------

    auto kernelBuilder = [&](auto /*fn object*/ kernelTaskBody) {
        auto taskOp = builder.create<vpux::VPURT::TaskOp>(builder.getUnknownLoc(), waitBarrier, updateBarrier);

        mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
        auto& block = taskOp.body().emplaceBlock();
        builder.setInsertionPointToStart(&block);

        kernelTaskBody();

        builder.restoreInsertionPoint(lastInsertionPoint);
    };

    kernelBuilder([&]() {
        const int64_t tile = checked_cast<int64_t>(cluster);

        auto swKernelOp = builder.create<VPUIP::SwKernelOp>(builder.getUnknownLoc(), inputCMX.buffer(),
                                                            outputCMX.buffer(), builtInFunction, getIntAttr(ctx, tile));
        VPUIP::initSwKernel(swKernelOp, inputCMX.buffer(), outputCMX.buffer(), kernelInfo.args, log);
    });
}

}  // namespace hwtest
}  // namespace vpux
