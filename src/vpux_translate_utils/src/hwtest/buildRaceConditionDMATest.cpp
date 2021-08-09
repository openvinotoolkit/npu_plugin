// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildRaceConditionDMATest(const nb::TestCaseJsonDescriptor&, mlir::ModuleOp module, mlir::OpBuilder builder,
                               Logger& log, mlir::Type inputType, mlir::Type outputType) {
    llvm::SmallVector<std::int64_t> in_shape{1, 16, 16, 16};
    llvm::SmallVector<std::int64_t> out_shape{1, 16, 16, 16};

    const auto totalsize = totalTensorSize(out_shape, outputType);

    const auto OUTPUT_0_CMX_OFFSET = 0;
    const auto OUTPUT_1_CMX_OFFSET = OUTPUT_0_CMX_OFFSET + totalsize;

    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(in_shape));
    const auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    const auto inType = mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, memSpaceAttr_in);

    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(out_shape));
    const auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    const auto outType = mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, memSpaceAttr_out);

    const auto funcType = builder.getFunctionType(makeArrayRef(std::vector<mlir::Type>{inType, outType, outType}),
                                                  makeArrayRef(std::vector<mlir::Type>{outType, outType}));

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             llvm::formatv("race_condition_dma_{0}_{1}", inputType, outputType).str(),
                                             funcType, builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcinput = func.getArgument(0);
    const auto funcoutput_0 = func.getArgument(1);
    const auto funcoutput_1 = func.getArgument(2);

    const auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto inputcmx_type =
            mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, inputcmx_memSpaceAttr);

    auto output_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_0_CMX_OFFSET);

    auto output_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_1_CMX_OFFSET);

    VPUIP::ConfigureBarrierOp lastBarrier;
    for (std::size_t i = 0; i < 256; ++i) {
        auto updateBarrier = funcBuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), i);

        funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput, output_0.getOperation()->getResult(0),
                                           i == 0 ? mlir::ValueRange() : mlir::ValueRange(lastBarrier.barrier()),
                                           mlir::ValueRange(updateBarrier.barrier()), false);

        funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput, output_1.getOperation()->getResult(0),
                                           i == 0 ? mlir::ValueRange() : mlir::ValueRange(lastBarrier.barrier()),
                                           mlir::ValueRange(updateBarrier.barrier()), false);

        lastBarrier = updateBarrier;
    }

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), output_0.getOperation()->getResult(0), funcoutput_0,
                                       mlir::ValueRange(lastBarrier.barrier()), mlir::ValueRange(), false);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), output_1.getOperation()->getResult(0), funcoutput_1,
                                       mlir::ValueRange(lastBarrier.barrier()), mlir::ValueRange(), false);

    funcBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{funcoutput_0, funcoutput_1});

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC, nullptr),
                getTensorType(out_shape, outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
