//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <functional>
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
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

    auto input = testDesc.getInputLayer();
    auto output = testDesc.getOutputLayer();
    auto iterationCount = testDesc.getIterationCount();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildRaceConditionDMATest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildRaceConditionDMATest: Got empty outputShape");

    const auto OUTPUT_0_CMX_OFFSET = 0;
    const auto OUTPUT_1_CMX_OFFSET = 0;

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    const auto funcType = builder.getFunctionType(makeArrayRef(std::vector<mlir::Type>{inType, outType, outType}),
                                                  makeArrayRef(std::vector<mlir::Type>{outType, outType}));

    auto func =
            builder.create<mlir::FuncOp>(loc, llvm::formatv("race_condition_dma_{0}_{1}", inputType, outputType).str(),
                                         funcType, builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcInput = func.getArgument(0);
    const auto funcOutput_0 = func.getArgument(1);
    const auto funcOutput_1 = func.getArgument(2);

    const auto outputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, outShape, outputType, DimsOrder::NHWC);

    auto output_0 = funcBuilder.create<VPURT::DeclareBufferOp>(loc, outputCMXType, VPURT::BufferSection::CMX_NN, 0,
                                                               OUTPUT_0_CMX_OFFSET);

    auto output_1 = funcBuilder.create<VPURT::DeclareBufferOp>(loc, outputCMXType, VPURT::BufferSection::CMX_NN, 1,
                                                               OUTPUT_1_CMX_OFFSET);

    VPURT::ConfigureBarrierOp lastBarrier;
    for (std::size_t i = 0; i < iterationCount; ++i) {
        auto updateBarrier = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, i);
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                funcBuilder, i == 0 ? mlir::ValueRange() : mlir::ValueRange(lastBarrier.barrier()),
                mlir::ValueRange(updateBarrier.barrier()), loc, funcInput, output_0.getOperation()->getResult(0), 0,
                false, false);

        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                funcBuilder, i == 0 ? mlir::ValueRange() : mlir::ValueRange(lastBarrier.barrier()),
                mlir::ValueRange(updateBarrier.barrier()), loc, funcInput, output_1.getOperation()->getResult(0), 1,
                false, false);

        lastBarrier = updateBarrier;
    }

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(lastBarrier.barrier()),
                                                mlir::ValueRange(), loc, output_0.getOperation()->getResult(0),
                                                funcOutput_0, 0, false, false);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(lastBarrier.barrier()),
                                                mlir::ValueRange(), loc, output_1.getOperation()->getResult(0),
                                                funcOutput_1, 1, false, false);

    funcBuilder.create<mlir::ReturnOp>(loc, mlir::ValueRange{funcOutput_0, funcOutput_1});

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
