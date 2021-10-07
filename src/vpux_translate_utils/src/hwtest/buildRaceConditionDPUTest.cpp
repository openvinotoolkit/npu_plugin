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

void buildRaceConditionDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                               mlir::Type outputType) {
    llvm::SmallVector<std::int64_t> in_shape{1, 16, 16, 16};
    llvm::SmallVector<std::int64_t> out_shape{1, 16, 16, 16};
    llvm::SmallVector<std::int64_t> wt_data_shape{16, 16, 1, 1};
    llvm::SmallVector<std::int64_t> wtTbl_data_shape{wt_data_shape[0], 1, 1, 4};

    const auto output_totalsize = totalTensorSize(out_shape, outputType);
    const auto input_totalsize = totalTensorSize(in_shape, inputType);
    const auto weights_totalsize = totalTensorSize(wt_data_shape, weightsType);
    const auto weightsTable_totalsize = 4 * wtTbl_data_shape[0] * wtTbl_data_shape[3];

    const auto OUTPUT_0_CMX_OFFSET = 0;
    const auto OUTPUT_1_CMX_OFFSET = OUTPUT_0_CMX_OFFSET + output_totalsize;
    const auto INPUT_0_CMX_OFFSET = OUTPUT_1_CMX_OFFSET + output_totalsize;
    const auto INPUT_1_CMX_OFFSET = INPUT_0_CMX_OFFSET + input_totalsize;
    const auto WEIGHTSTABLE_0_CMX_OFFSET = INPUT_1_CMX_OFFSET + input_totalsize;
    const auto WEIGHTSTABLE_1_CMX_OFFSET = WEIGHTSTABLE_0_CMX_OFFSET + weightsTable_totalsize;
    const auto WEIGHTS_0_CMX_OFFSET = WEIGHTSTABLE_1_CMX_OFFSET + weightsTable_totalsize;
    const auto WEIGHTS_1_CMX_OFFSET = WEIGHTS_0_CMX_OFFSET + weights_totalsize;

    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(in_shape));
    const auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    const auto inType = mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, memSpaceAttr_in);

    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(in_shape));
    const auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    const auto outType = mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, memSpaceAttr_out);

    const auto funcType = builder.getFunctionType(makeArrayRef(std::vector<mlir::Type>{inType, outType, outType}),
                                                  makeArrayRef(std::vector<mlir::Type>{outType, outType}));

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             llvm::formatv("race_condition_dpu_{0}_{1}", inputType, outputType).str(),
                                             funcType, builder.getStringAttr("private"));

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcinput = func.getArgument(0);
    const auto funcoutput_0 = func.getArgument(1);
    const auto funcoutput_1 = func.getArgument(2);

    const auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto inputcmx_type =
            mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, inputcmx_memSpaceAttr);

    const auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto outputcmx_type =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, outputcmx_memSpaceAttr);

    const auto weightDataAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wt_data_shape));
    const auto weight_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightData_ddr_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType,
                                                           weightDataAffineMaps, weight_data_ddr_memSpaceAttr);

    const auto wtData_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto wtData_cmx_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps,
                                                       wtData_cmx_memSpaceAttr);

    const auto weightTbl_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightTblAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wtTbl_data_shape));
    const auto weightTblData_ddr_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, true), weightTblAffineMaps,
                                  weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, true));
    const std::vector<int32_t> wtTbl_0_data_values_vec =
            generateWeightsTablesValues(testDesc, WEIGHTS_0_CMX_OFFSET, inputcmx_type, outputcmx_type, wtData_cmx_type);
    const std::vector<int32_t> wtTbl_1_data_values_vec =
            generateWeightsTablesValues(testDesc, WEIGHTS_1_CMX_OFFSET, inputcmx_type, outputcmx_type, wtData_cmx_type);
    const auto wtTbl_0_data_values = makeArrayRef<int32_t>(wtTbl_0_data_values_vec);
    const auto wtTbl_1_data_values = makeArrayRef<int32_t>(wtTbl_1_data_values_vec);
    const auto wtTbl_0_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_0_data_values);
    const auto wtTbl_1_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_1_data_values);
    const auto wt_data_vals = generateWeights(wt_data_shape, weightsType, builder.getContext(), "weights.dat");

    const auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    const auto wtTbl_cmx_type = mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, true),
                                                      weightTblAffineMaps, wtTbl_cmx_memSpaceAttr);

    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto weight_data_ddr = funcBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightData_ddr_type,
                                                                wt_data_attr.reorder(DimsOrder::NHWC));

    auto weightTbl_0_data_ddr =
            funcBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_0_data_vals).reorder(DimsOrder::NHWC));

    auto weightTbl_1_data_ddr =
            funcBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_1_data_vals).reorder(DimsOrder::NHWC));

    auto wtData_cmx_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtData_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, WEIGHTS_0_CMX_OFFSET);

    auto wtData_cmx_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtData_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, WEIGHTS_1_CMX_OFFSET);

    auto wtTbl_cmx_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtTbl_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, WEIGHTSTABLE_0_CMX_OFFSET);

    auto wtTbl_cmx_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), wtTbl_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, WEIGHTSTABLE_1_CMX_OFFSET);

    auto parent_input_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_0_CMX_OFFSET);

    auto parent_input_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, INPUT_0_CMX_OFFSET);

    auto input_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                              VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_0_CMX_OFFSET);

    auto input_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                              VPUIP::MemoryLocation::VPU_CMX_NN, 1, INPUT_1_CMX_OFFSET);

    auto parent_output_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_0_CMX_OFFSET);

    auto parent_output_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, OUTPUT_1_CMX_OFFSET);

    auto output_0 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_0_CMX_OFFSET);

    auto output_1 = funcBuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 1, OUTPUT_1_CMX_OFFSET);

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
                actChannelLength, /*odu_permutation=*/nullptr);

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

    auto lastBarrier = funcBuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput, input_0.getOperation()->getResult(0),
                                       mlir::ValueRange(), mlir::ValueRange(lastBarrier.barrier()), false);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput, input_1.getOperation()->getResult(0),
                                       mlir::ValueRange(), mlir::ValueRange(lastBarrier.barrier()), false);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), weight_data_ddr.getOperation()->getResult(0),
                                       wtData_cmx_0.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(lastBarrier.barrier()), false);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), weight_data_ddr.getOperation()->getResult(0),
                                       wtData_cmx_1.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(lastBarrier.barrier()), false);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), weightTbl_0_data_ddr.getOperation()->getResult(0),
                                       wtTbl_cmx_0.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(lastBarrier.barrier()), false);

    funcBuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), weightTbl_1_data_ddr.getOperation()->getResult(0),
                                       wtTbl_cmx_1.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(lastBarrier.barrier()), false);

    for (std::size_t i = 1; i < 256; ++i) {
        auto updateBarrier = funcBuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), i);

        createDPUOp(wtData_cmx_0, wtTbl_cmx_0, input_0, output_0, parent_input_0, parent_output_0, lastBarrier,
                    updateBarrier);
        createDPUOp(wtData_cmx_1, wtTbl_cmx_1, input_1, output_1, parent_input_1, parent_output_1, lastBarrier,
                    updateBarrier);

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
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC),
                getTensorType(out_shape, outputType, DimsOrder::NHWC)});
}

}  // namespace hwtest
}  // namespace vpux
