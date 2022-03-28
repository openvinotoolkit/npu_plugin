//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/hwtests_ops.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

namespace {

using opBuilderCallback =
        std::function<void(const nb::TestCaseJsonDescriptor&, mlir::ModuleOp, mlir::OpBuilder, Logger&,
                           SmallVector<mlir::Type>, vpux::VPURT::DeclareBufferOp, vpux::VPURT::DeclareBufferOp,
                           mlir::ValueRange, mlir::ValueRange, size_t, size_t)>;
opBuilderCallback getOpBuilder(nb::CaseType caseType) {
    switch (caseType) {
    case nb::CaseType::ActShave:
        return &buildActShaveTask;
    default:
        VPUX_THROW("ActShave case is only supported for generic RaceCondition");
    }
}

}  // namespace

/*
//                       [input]
//           (dma to CMX0)     (dma to CMX1)
//                |                 |
//              /   \             /   \
//            |       |         |       |
//          (op)    (op)       (op)    (op)
//            |       |         |       |
//          (op)    (op)       (op)    (op)
//            |       |         |       |
//        ... (loop with conv ops and barriers)
//            |       |         |       |
//          (op)    (op)       (op)    (op)
//            |       |         |       |
//       [output0]  [output1] [output2]  [output3]
*/

void buildRaceConditionTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto raceConditionParams = testDesc.getRaceConditionParams();
    auto testDescUnderlyingOp = testDesc.getUnderlyingOp();

    auto underlyingOpBuilder = getOpBuilder(testDescUnderlyingOp->getCaseType());

    auto input = testDescUnderlyingOp->getInputLayer();
    auto output = testDescUnderlyingOp->getOutputLayer();
    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    SmallVector<mlir::Type> inputTypes;
    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    inputTypes.push_back(inputParamType);
    SmallVector<mlir::Type> outputTypes;
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    const auto outputsCount = raceConditionParams.requestedClusters * raceConditionParams.requestedUnits;
    inputTypes.insert(inputTypes.end(), outputsCount, outputParamType);
    outputTypes.insert(outputTypes.end(), outputsCount, outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), makeArrayRef(outputTypes));

    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("race_confition_{0}_{1}_{2}", testDescUnderlyingOp->getCaseStr(), inputType, outputType)
                    .str(),
            funcType, builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto funcInput = func.getArgument(0);

    const auto inputCMXOffset = 0;
    const auto outputCMXOffset = inputCMXOffset + totalTensorSize(inShape, inputType);

    auto inputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, inShape, inputType, DimsOrder::NHWC);

    auto outputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, outShape, outputType, DimsOrder::NHWC);

    VPURT::ConfigureBarrierOp waitBarrier;
    size_t barrierNumber = 0;
    SmallVector<mlir::Value> waitBarriers;
    SmallVector<vpux::VPURT::DeclareBufferOp> outputs;
    SmallVector<mlir::Type> cnnOpOutputs;
    for (size_t cluster = 0; cluster < raceConditionParams.requestedClusters; ++cluster) {
        auto inputDataDMA = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

        auto inputCMX = createDeclareTensorOp(funcBuilder, inputCMXType, VPURT::BufferSection::CMX_NN,
                                              static_cast<int>(cluster), inputCMXOffset);

        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(),
                                                    mlir::ValueRange(inputDataDMA.barrier()), builder.getUnknownLoc(),
                                                    funcInput, getTensorResult(inputCMX), cluster, false, false);

        auto localOutputCMXOffset = outputCMXOffset;
        for (size_t unit = 0; unit < raceConditionParams.requestedUnits; ++unit) {
            auto outputCMX = createDeclareTensorOp(funcBuilder, outputCMXType, VPURT::BufferSection::CMX_NN,
                                                   static_cast<int>(cluster), localOutputCMXOffset);
            outputs.emplace_back(outputCMX);
            cnnOpOutputs.emplace_back(getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr));
            localOutputCMXOffset += totalTensorSize(outShape, outputType);
            for (size_t iter = 0; iter < raceConditionParams.iterationsCount; ++iter) {
                auto updateBarrier =
                        funcBuilder.create<VPURT::ConfigureBarrierOp>(funcBuilder.getUnknownLoc(), barrierNumber++);
                underlyingOpBuilder(
                        *testDescUnderlyingOp, module, funcBuilder, log, inputTypes, inputCMX, outputCMX,
                        iter == 0 ? mlir::ValueRange(inputDataDMA.barrier()) : mlir::ValueRange(waitBarrier.barrier()),
                        mlir::ValueRange(updateBarrier.barrier()), cluster, unit);
                waitBarrier = updateBarrier;
            }
            waitBarriers.emplace_back(waitBarrier.barrier());
        }
    }

    SmallVector<mlir::BlockArgument> returnOps;
    for (size_t i = 0; i < outputs.size(); ++i) {
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(llvm::makeArrayRef(waitBarriers)),
                                                    mlir::ValueRange(), builder.getUnknownLoc(),
                                                    getTensorResult(outputs[i]),
                                                    func.getArgument(static_cast<unsigned int>(i + 1)));
        returnOps.emplace_back(func.getArgument(static_cast<unsigned int>(i + 1)));
    }

    funcBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                       mlir::ValueRange(ArrayRef<mlir::BlockArgument>(returnOps)));

    //  Pass Manager
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::ReferenceHW,
                                           static_cast<int>(raceConditionParams.requestedClusters), log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    //  CNN Operation
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               cnnOpOutputs);
}

}  // namespace hwtest
}  // namespace vpux
