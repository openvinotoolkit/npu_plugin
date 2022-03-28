//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildDMA(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder, Logger& log,
              mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayer();
    auto dmaParams = testDesc.getDMAparams();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildDMA: Input rank is 0");
    VPUX_THROW_UNLESS(inShape == outShape, "buildDMA: in_shape and out_shape don't match");
    VPUX_THROW_UNLESS(inputType == outputType, "buildDMA: outputType and outputType don't match");

    auto inputTotalSize = totalTensorSize(inShape, inputType);
    auto outputTotalSize = totalTensorSize(outShape, outputType);

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC));

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("dma_from_{0}_{1}_to_{2}_{3}", nb::to_string(dmaParams.srcLocation), inputType,
                          nb::to_string(dmaParams.dstLocation), outputType)
                    .str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcInput0 = func.getArgument(0);
    auto funcOutput = func.getArgument(1);

    // input - output tensors

    llvm::SmallVector<mlir::Value> waitBarriers;
    mlir::Value DMAinput;
    size_t CMX0_AVIABLE_OFFSET = 0;
    size_t CMX1_AVIABLE_OFFSET = 0;
    int barrierNumber = 0;
    if (dmaParams.srcLocation == nb::MemoryLocation::DDR) {
        DMAinput = funcInput0;
    } else if (dmaParams.srcLocation == nb::MemoryLocation::CMX0 || dmaParams.srcLocation == nb::MemoryLocation::CMX1) {
        auto inputCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, inShape, inputType, DimsOrder::NHWC);
        auto inputCMX = createDeclareTensorOp(
                funcbuilder, inputCMXtype, VPURT::BufferSection::CMX_NN,
                dmaParams.srcLocation == nb::MemoryLocation::CMX0 ? 0 : 1,
                dmaParams.srcLocation == nb::MemoryLocation::CMX0 ? CMX0_AVIABLE_OFFSET : CMX1_AVIABLE_OFFSET);
        if (dmaParams.srcLocation == nb::MemoryLocation::CMX0) {
            CMX0_AVIABLE_OFFSET += inputTotalSize;
        } else {
            CMX1_AVIABLE_OFFSET += inputTotalSize;
        }
        auto barrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier.barrier()),
                                              builder.getUnknownLoc(), funcInput0,
                                              inputCMX.getOperation()->getResult(0));
        waitBarriers.emplace_back(barrier.barrier());
        DMAinput = inputCMX.getOperation()->getResult(0);
    } else {
        VPUX_THROW("Unsupported src memory location {0}", nb::to_string(dmaParams.srcLocation));
    }
    mlir::Value DMAoutput;
    if (dmaParams.dstLocation == nb::MemoryLocation::DDR) {
        DMAoutput = funcOutput;
    } else if (dmaParams.dstLocation == nb::MemoryLocation::CMX0 || dmaParams.dstLocation == nb::MemoryLocation::CMX1) {
        auto outputCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, outShape, outputType, DimsOrder::NHWC);
        auto outputCMX = createDeclareTensorOp(
                funcbuilder, outputCMXtype, VPURT::BufferSection::CMX_NN,
                dmaParams.dstLocation == nb::MemoryLocation::CMX0 ? 0 : 1,
                dmaParams.dstLocation == nb::MemoryLocation::CMX0 ? CMX0_AVIABLE_OFFSET : CMX1_AVIABLE_OFFSET);
        if (dmaParams.dstLocation == nb::MemoryLocation::CMX0) {
            CMX0_AVIABLE_OFFSET += outputTotalSize;
        } else {
            CMX1_AVIABLE_OFFSET += outputTotalSize;
        }
        DMAoutput = outputCMX.getOperation()->getResult(0);
    } else {
        VPUX_THROW("Unsupported dst memory location {0}", nb::to_string(dmaParams.dstLocation));
    }

    VPURT::ConfigureBarrierOp DMAtaskBarrier;
    if (dmaParams.dstLocation == nb::MemoryLocation::CMX0 || dmaParams.dstLocation == nb::MemoryLocation::CMX1) {
        DMAtaskBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    }
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(llvm::ArrayRef<mlir::Value>(waitBarriers)),
                                          dmaParams.dstLocation == nb::MemoryLocation::DDR
                                                  ? mlir::ValueRange()
                                                  : mlir::ValueRange(DMAtaskBarrier.barrier()),
                                          builder.getUnknownLoc(), DMAinput, DMAoutput, dmaParams.engine, false, false);

    if (dmaParams.dstLocation == nb::MemoryLocation::CMX0 || dmaParams.dstLocation == nb::MemoryLocation::CMX1) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(DMAtaskBarrier.barrier()),
                                              mlir::ValueRange(), builder.getUnknownLoc(), DMAoutput, funcOutput);
    }

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcOutput);
    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
