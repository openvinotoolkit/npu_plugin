// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildPipeline(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType,
                   bool isSequential) {
    llvm::SmallVector<std::int64_t> in_shape{1, 16, 16, 16};
    llvm::SmallVector<std::int64_t> out_shape{1, 16, 16, 16};
    llvm::SmallVector<std::int64_t> wt_data_shape{16, 16, 1, 1};
    llvm::SmallVector<std::int64_t> wtTbl_data_shape{wt_data_shape[0], 1, 1, 4};

    auto weight = testDesc.getWeightLayer();

    const auto output_totalsize = totalTensorSize(out_shape, outputType);
    const auto input_totalsize = totalTensorSize(in_shape, inputType);
    const auto weightsTable_totalsize = 4 * wtTbl_data_shape[0] * wtTbl_data_shape[3];

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;
    const auto WEIGHTS_CMX_OFFSET = WEIGHTSTABLE_CMX_OFFSET + weightsTable_totalsize;

    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(in_shape));

    const auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    const auto inType = mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, memSpaceAttr_in);

    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(out_shape));

    const auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    const auto outType = mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, memSpaceAttr_out);

    const auto funcType = builder.getFunctionType(makeArrayRef(std::vector<mlir::Type>{inType, outType}),
                                                  makeArrayRef(std::vector<mlir::Type>{outType}));

    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("pipeline_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcinput = func.getArgument(0);
    const auto funcoutput = func.getArgument(1);

    // input - output cmx tensors
    const auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto inputcmx_type =
            mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, inputcmx_memSpaceAttr);

    const auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto outputcmx_type =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, outputcmx_memSpaceAttr);

    // weights data
    const auto weight_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightDataAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wt_data_shape));
    const auto weightData_ddr_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType,
                                                           weightDataAffineMaps, weight_data_ddr_memSpaceAttr);

    // get weights from a file
    const auto wt_data_vals = generateWeights(wt_data_shape, weightsType, builder.getContext(), "weights.dat");
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto weight_data_ddr = funcBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightData_ddr_type,
                                                                wt_data_attr.reorder(DimsOrder::NHWC));

    // weights cmx tensor
    const auto wtData_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto wtData_cmx_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps,
                                                       wtData_cmx_memSpaceAttr);
    auto wtData_cmx_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtData_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, WEIGHTS_CMX_OFFSET);

    auto wtData_cmx_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtData_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, WEIGHTS_CMX_OFFSET);

    // weights table ddr tensor
    const auto weightTbl_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightTblAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wtTbl_data_shape));
    const auto weightTblData_ddr_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, true), weightTblAffineMaps,
                                  weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, true));

    const std::vector<int32_t> wtTbl_data_values_vec =
            generateWeightsTablesValues(testDesc, WEIGHTS_CMX_OFFSET, inputcmx_type, outputcmx_type, wtData_cmx_type);
    const auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    const auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    const auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto wtTbl_cmx_type = mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, true),
                                                      weightTblAffineMaps, wtTbl_cmx_memSpaceAttr);
    auto wtTbl_cmx_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtTbl_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, WEIGHTSTABLE_CMX_OFFSET);
    auto wtTbl_cmx_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtTbl_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, WEIGHTSTABLE_CMX_OFFSET);

    auto createDPUOp = [&](auto& wtData_cmx, auto& wtTbl_cmx, auto& inputcmx, auto& outputcmx, auto& parent_inputcmx,
                           auto& parent_outputcmx, auto& barrier0, auto& barrier1) {
        // NCE Task
        std::vector<int32_t> stried_vec{1, 1};
        const auto strides = getIntArrayAttr(builder, stried_vec);
        std::vector<int32_t> padding_vec{0, 0, 0, 0};
        const auto kernel_padding = getIntArrayAttr(builder, padding_vec);
        SmallVector<int64_t> kernel_vec = {wt_data_shape[2], wt_data_shape[3]};
        const auto kernel_size = getIntArrayAttr(builder, kernel_vec);
        mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);

        auto nceTask = funcBuilder.create<VPUIP::NCEClusterTaskOp>(
                builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
                wtData_cmx.getOperation()->getResult(0), wtTbl_cmx.getOperation()->getResult(0), nullptr,
                parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
                outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
                mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::CONV, kernel_size, strides, kernel_padding,
                actChannelLength, /*odu_permutation=*/nullptr, /*weights_plt*/ nullptr);

        nceTask.addPPETask(funcBuilder);

        // Create DPU task for NCE task
        nceTask.variants().emplaceBlock();
        auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

        auto start = getIntArrayAttr(builder, std::vector<std::int32_t>{0, 0, 0});
        auto end = getIntArrayAttr(builder, std::vector<std::int32_t>{15, 15, 15});
        auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, 0), getIntAttr(builder, 0), getIntAttr(builder, 0),
                                           getIntAttr(builder, 0), builder.getContext());

        variantbuilder.template create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), nullptr, start, end, pad,
                                                         VPUIP::MPEMode::CUBOID_16x16);
    };

    auto barrierIndex = 0;
    Optional<VPUIP::ConfigureBarrierOp> lastBarrier;

    auto generatePipeline = [&](int localeIndex, bool last = false) {
        auto inputcmx = funcBuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                                   VPUIP::MemoryLocation::VPU_CMX_NN, localeIndex,
                                                                   INPUT_CMX_OFFSET);

        auto outputcmx = funcBuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), outputcmx_type,
                                                                    VPUIP::MemoryLocation::VPU_CMX_NN, localeIndex,
                                                                    OUTPUT_CMX_OFFSET);

        auto barrier0 = funcBuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierIndex++);

        funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput, inputcmx.getOperation()->getResult(0),
                                           isSequential && lastBarrier.hasValue()
                                                   ? mlir::ValueRange(lastBarrier.getValue().barrier())
                                                   : mlir::ValueRange(),
                                           mlir::ValueRange(barrier0.barrier()), false);

        auto& wtData_cmx = localeIndex == 0 ? wtData_cmx_0 : wtData_cmx_1;

        funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), weight_data_ddr.getOperation()->getResult(0),
                                           wtData_cmx.getOperation()->getResult(0), mlir::ValueRange(),
                                           mlir::ValueRange(barrier0.barrier()), false);

        auto& wtTbl_cmx = localeIndex == 0 ? wtTbl_cmx_0 : wtTbl_cmx_1;

        funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), weightTbl_data_ddr.getOperation()->getResult(0),
                                           wtTbl_cmx.getOperation()->getResult(0), mlir::ValueRange(),
                                           mlir::ValueRange(barrier0.barrier()), false);

        auto parent_inputcmx = funcBuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                                          VPUIP::MemoryLocation::VPU_CMX_NN,
                                                                          localeIndex, INPUT_CMX_OFFSET);

        auto parent_outputcmx = funcBuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), outputcmx_type,
                                                                           VPUIP::MemoryLocation::VPU_CMX_NN,
                                                                           localeIndex, OUTPUT_CMX_OFFSET);

        auto barrier1 = funcBuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierIndex++);

        createDPUOp(wtData_cmx_0, wtTbl_cmx_0, inputcmx, outputcmx, parent_inputcmx, parent_outputcmx, barrier0,
                    barrier1);

        if (isSequential && !last) {
            lastBarrier = funcBuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierIndex++);
        }

        funcBuilder.create<VPUIP::NNDMAOp>(
                builder.getUnknownLoc(), outputcmx, funcoutput, mlir::ValueRange(barrier1.barrier()),
                isSequential && !last ? mlir::ValueRange(lastBarrier.getValue().barrier()) : mlir::ValueRange(), false);
    };

    generatePipeline(0);
    generatePipeline(1);
    generatePipeline(0);
    generatePipeline(1);
    generatePipeline(0, true);

    funcBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC)});
}

}  // namespace hwtest
}  // namespace vpux
