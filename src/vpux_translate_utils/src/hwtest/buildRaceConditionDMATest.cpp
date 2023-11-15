//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildRaceConditionDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayerList().front();
    auto output = testDesc.getOutputLayers().front();
    auto iterationCount = testDesc.getIterationCount();
    const auto numClusters = testDesc.getNumClusters();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildRaceConditionDMATest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildRaceConditionDMATest: Got empty outputShape");

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    SmallVector<mlir::Type> inputTypes(numClusters, outType);
    inputTypes.insert(inputTypes.begin(), inType);

    SmallVector<mlir::Type> outputTypes(numClusters, outType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), makeArrayRef(outputTypes));

    auto func =
            builder.create<mlir::func::FuncOp>(loc, printToString("race_condition_dma_{0}_{1}", inputType, outputType),
                                               funcType, builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcInput = func.getArgument(0);
    SmallVector<mlir::BlockArgument> funcOutputs;

    for (std::size_t idx = 1; idx <= numClusters; ++idx) {
        funcOutputs.push_back(func.getArgument(idx));
    }

    SmallVector<mlir::MemRefType> outputCMXTypes;
    SmallVector<VPURT::DeclareBufferOp> outputs;

    for (std::size_t idx = 0; idx < numClusters; ++idx) {
        outputCMXTypes.push_back(
                getMemRefType(VPURT::BufferSection::CMX_NN, idx, outShape, outputType, DimsOrder::NHWC));
        outputs.push_back(funcBuilder.create<VPURT::DeclareBufferOp>(
                loc, outputCMXTypes[idx], VPURT::BufferSection::CMX_NN, idx, /*byteOffset=*/0));
    }

    VPURT::ConfigureBarrierOp lastBarrier;
    for (std::size_t i = 0; i < iterationCount; ++i) {
        auto updateBarrier = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, i);

        for (std::size_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx += 2) {
            vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                    funcBuilder, i == 0 ? mlir::ValueRange() : mlir::ValueRange(lastBarrier.getBarrier()),
                    mlir::ValueRange(updateBarrier.getBarrier()), loc, funcInput,
                    outputs[clusterIdx].getOperation()->getResult(0), 0);

            vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                    funcBuilder, i == 0 ? mlir::ValueRange() : mlir::ValueRange(lastBarrier.getBarrier()),
                    mlir::ValueRange(updateBarrier.getBarrier()), loc, funcInput,
                    outputs[clusterIdx + 1].getOperation()->getResult(0), 1);
        }
        lastBarrier = updateBarrier;
    }

    for (std::size_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx += 2) {
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                funcBuilder, mlir::ValueRange(lastBarrier.getBarrier()), mlir::ValueRange(), loc,
                outputs[clusterIdx].getOperation()->getResult(0), funcOutputs[clusterIdx], 0);
        if (clusterIdx + 1 < numClusters) {
            vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                    funcBuilder, mlir::ValueRange(lastBarrier.getBarrier()), mlir::ValueRange(), loc,
                    outputs[clusterIdx].getOperation()->getResult(0), funcOutputs[clusterIdx + 1], 1);
        }
    }

    auto outputsRef = makeArrayRef(funcOutputs);
    funcBuilder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange(outputsRef));

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);

    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW,
                                           /*numOfDPUGroups=*/numClusters, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    SmallVector<mlir::Type> userOutputs(numClusters,
                                        getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr));
    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               makeArrayRef(userOutputs));
}

}  // namespace hwtest
}  // namespace vpux
