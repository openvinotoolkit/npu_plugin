//
// Copyright 2022 Intel Corporation.
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

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/hwtests_ops.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace hwtest {

void buildReadAfterWriteACTDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int8 = builder.getIntegerType(8, true);
    const auto overwritingType = int8;
    const auto rewritableType = overwritingType;

    const auto input = testDesc.getInputLayer();
    const auto output = testDesc.getOutputLayer();
    const auto iterationCount = testDesc.getIterationCount();
    const auto cluster = testDesc.getClusterNumber();

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> overwritingShape{1, 1, 1, 1};
    const auto rewritableShape = overwritingShape;

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildReadAfterWriteACTDMATest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildReadAfterWriteACTDMATest: Got empty outputShape");

    const auto inputCMXSize = vpux::hwtest::totalTensorSize(outputShape, inputType);
    const auto overwritingCMXSize = vpux::hwtest::totalTensorSize(overwritingShape, overwritingType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);
    const auto rewritableInputCMXSize = overwritingCMXSize;

    const auto OVERWRITING_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = 0;
    auto REWRITABLE_INPUT_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize - rewritableInputCMXSize;
    auto OUTPUT_CMX_OFFSET = REWRITABLE_INPUT_CMX_OFFSET + rewritableInputCMXSize;

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);
    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(inputParamType);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            loc, llvm::formatv("read_after_write_act_dma_{0}_{1}", inputType, outputType).str(), funcType,
            builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          DimsOrder::NHWC, cluster, INPUT_CMX_OFFSET);
    auto overwritingCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, overwritingShape,
                                                overwritingType, DimsOrder::NHWC, cluster, OVERWRITING_CMX_OFFSET);
    auto rewritableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, rewritableShape,
                                               rewritableType, DimsOrder::NHWC, cluster, REWRITABLE_INPUT_CMX_OFFSET);
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           DimsOrder::NHWC, cluster, OUTPUT_CMX_OFFSET);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 0);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                          mlir::ValueRange(updateBarrier.barrier()), loc, functionInput,
                                          inputCMX.getOperation()->getResult(0));

    auto waitBarrier = updateBarrier;
    for (std::size_t i = 1; i + 1 < iterationCount; i += 2) {
        if (i != 1) {
            inputCMX = outputCMX;
            OUTPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
            outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                              DimsOrder::NHWC, cluster, OUTPUT_CMX_OFFSET);

            REWRITABLE_INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET - rewritableInputCMXSize;
            rewritableCMX =
                    createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, rewritableShape,
                                          rewritableType, DimsOrder::NHWC, cluster, REWRITABLE_INPUT_CMX_OFFSET);
        }
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i);
        buildActShaveTask(testDesc, module, functionBuilder, log, inputTypes, inputCMX, outputCMX,
                          mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(updateBarrier.barrier()), cluster);
        waitBarrier = updateBarrier;

        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i + 1);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()),
                                              mlir::ValueRange(updateBarrier.barrier()), loc,
                                              overwritingCMX.getOperation()->getResult(0),
                                              rewritableCMX.getOperation()->getResult(0), cluster, false, false);
        waitBarrier = updateBarrier;
    }

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(),
                                          loc, outputCMX.getOperation()->getResult(0), functionOutput);

    functionBuilder.create<mlir::ReturnOp>(loc, mlir::ValueRange{functionOutput});

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
