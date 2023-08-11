//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"

#include "vpux/utils/core/profiling.hpp"

using namespace vpux;

namespace {

//
//  CaptureWorkpointPass
//

class CaptureWorkpointPass final : public VPUIP::CaptureWorkpointBase<CaptureWorkpointPass> {
public:
    explicit CaptureWorkpointPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void insertCaptureDma(mlir::OpBuilder& builder, int64_t profOutputId, size_t dstDdrOffset) {
    auto* ctx = builder.getContext();

    const auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::Register));
    const vpux::NDTypeInterface hwTimerType =
            getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, memKindAttr);

    const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, PROFILING_WORKPOINT_READ_ATTR));
    // Create declaration for source buffer which corresponds to HW register with free-running counter
    auto hwRegOp = builder.create<VPURT::DeclareBufferOp>(loc, hwTimerType, VPURT::BufferSection::Register,
                                                          VPUIP::HW_PLL_WORKPOINT_ABSOLUTE_ADDR);

    const auto profilingOutputType = mlir::MemRefType::get(hwTimerType.getShape().raw(), hwTimerType.getElementType());
    auto dstBufProfResultOp = builder.create<VPURT::DeclareBufferOp>(
            loc, profilingOutputType, VPURT::BufferSection::ProfilingOutput, profOutputId, dstDdrOffset);

    const auto port = 0;
    // Since the payload is copied into the final destination is DDR no barriers needed, so may be inserted anywhere in
    // the network without barriers setup
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, /*waitBarriers=*/{}, /*updateBarriers=*/{}, loc, hwRegOp.buffer(),
                                          dstBufProfResultOp.buffer(), port, /*is_out_of_order=*/true,
                                          /*is_critical=*/false, /*spillId=*/nullptr);
}

void CaptureWorkpointPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();
    const auto arch = VPU::getArch(module);
    VPUX_THROW_UNLESS(arch == VPU::ArchKind::VPUX37XX, "Capture of workpoint available only for VPUX37XX");

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(module, netOp, func);
    mlir::OpBuilder builder(&func.getBody().front().front());

    const auto profOutputId = static_cast<int64_t>(netOp.getProfilingOutputsCount());
    const auto outputResult = mlir::MemRefType::get({VPUIP::NUM_CAPTURED_WORKPOINTS}, getUInt32Type(ctx));

    // Update network output information to have also new pll profiling result
    auto profilingResult = addNewProfilingOutput(ctx, func, netOp, outputResult, "pll");
    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);
    returnOp.operandsMutable().append(profilingResult);

    // At 37XX workpoint register is accesible by DMA, so insert transactions, otherwise FW handle it
    bool isWorkpointDmaAcessible = arch == VPU::ArchKind::VPUX37XX;
    if (isWorkpointDmaAcessible) {
        builder.setInsertionPoint(&func.getBody().front().front());
        // Capture setup in the begin of inference
        insertCaptureDma(builder, profOutputId, 0);
        // And in the end
        builder.setInsertionPoint(returnOp);
        insertCaptureDma(builder, profOutputId, VPUIP::HW_PLL_WORKPOINT_SIZE);
    }
}

}  // namespace

//
// createCaptureWorkpointPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCaptureWorkpointPass(Logger log) {
    return std::make_unique<CaptureWorkpointPass>(log);
}
