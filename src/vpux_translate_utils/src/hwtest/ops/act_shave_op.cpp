//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include <vpux/hwtest/ops/act_shave_op.hpp>

#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"

namespace vpux {
namespace hwtest {

namespace {

VPUIP::KernelInfo getKernelInfo(nb::ActivationLayer activation, mlir::MLIRContext* ctx) {
    switch (activation.activationType) {
    case nb::ActivationType::HSwish:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"hswish_fp16"}, {"hswish_fp16.cpp"}};
    case nb::ActivationType::Sigmoid:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sigmoid_fp16"}, {"sigmoid_fp16.c"}};
    case nb::ActivationType::Softmax:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, activation.axis)},
                                 {"singleShaveSoftmax"},
                                 {"single_shave_softmax.cpp"}};
    case nb::ActivationType::vau_sigm:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_sigm_fp16"}, {"vau_sigm_fp16.c"}};
    case nb::ActivationType::vau_sqrt:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_sqrt_fp16"}, {"vau_sqrt_fp16.c"}};
    case nb::ActivationType::vau_tanh:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_tanh_fp16"}, {"vau_tanh_fp16.c"}};
    case nb::ActivationType::vau_log:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_log_fp16"}, {"vau_log_fp16.c"}};
    case nb::ActivationType::vau_exp:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_exp_fp16"}, {"vau_exp_fp16.c"}};
    case nb::ActivationType::lsu_b16:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"lsu_b16"}, {"lsu_b16.cpp"}};
    case nb::ActivationType::lsu_b16_vec:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"lsu_b16_vec"}, {"lsu_b16_vec.cpp"}};
    case nb::ActivationType::sau_dp4:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sau_dp4"}, {"sau_dp4.cpp"}};
    case nb::ActivationType::sau_dp4a:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sau_dp4a"}, {"sau_dp4a.cpp"}};
    case nb::ActivationType::sau_dp4m:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sau_dp4m"}, {"sau_dp4m.cpp"}};
    case nb::ActivationType::vau_dp4:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_dp4"}, {"vau_dp4.cpp"}};
    case nb::ActivationType::vau_dp4a:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_dp4a"}, {"vau_dp4a.cpp"}};
    case nb::ActivationType::vau_dp4m:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"vau_dp4m"}, {"vau_dp4m.cpp"}};
    default:
        VPUX_THROW("Activation is not supported for ActShave tests");
    }
}

}  // namespace

void buildActShaveTask(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                       Logger& log, ArrayRef<mlir::Type> inputTypes,
                       SmallVector<vpux::VPURT::DeclareBufferOp>& inputCMX, vpux::VPURT::DeclareBufferOp outputCMX,
                       mlir::ValueRange waitBarrier, mlir::ValueRange updateBarrier, size_t cluster, size_t /*unit*/) {
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

    SmallString builtInFunctionName{"builtin_actshave_"};

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

        SmallVector<mlir::Value> inputCMXValues;
        for (auto& input : inputCMX) {
            inputCMXValues.push_back(input.buffer());
        }

        auto swKernelOp = builder.create<VPUIP::SwKernelOp>(builder.getUnknownLoc(), mlir::ValueRange{inputCMXValues},
                                                            outputCMX.buffer(), builtInFunction, getIntAttr(ctx, tile));
        VPUIP::initSwKernel(swKernelOp, mlir::ValueRange{inputCMXValues}, outputCMX.buffer(), kernelInfo.args, log);
    });
}

}  // namespace hwtest
}  // namespace vpux
