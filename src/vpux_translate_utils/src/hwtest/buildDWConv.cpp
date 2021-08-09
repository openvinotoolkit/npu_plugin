//
// Copyright 2021 Intel Corporation.
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

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildDWConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                 Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto input = testDesc.getInputLayer();
    auto weight = testDesc.getWeightLayer();
    auto conv = testDesc.getConvLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(conv.group == in_shape[1],
                      "For Depthwise convolution group should be equal to no. of input channels");

    std::vector<int64_t> filter_size{weight.shape[2], weight.shape[3]};
    std::vector<int64_t> stried_vec(conv.stride.begin(), conv.stride.end());
    std::vector<int64_t> padding_vec(conv.pad.begin(), conv.pad.end());

    SmallVector<int64_t> wt_data_shape{weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]};

    const char* weight_file_name = "weight.dat";

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);
    auto weight_totalsize = totalTensorSize(wt_data_shape, weightsType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTS_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;
    const auto ACTIVATIONWINDOW_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weight_totalsize;

    SmallVector<mlir::Type> inputTypes;
    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(in_shape));
    auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    inputTypes.push_back(mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, memSpaceAttr_in));
    auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(out_shape));
    auto outputParamType =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, memSpaceAttr_out);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("dw_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data
    auto weight_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightDataAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wt_data_shape));
    auto weightData_ddr_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps,
                                                     weight_data_ddr_memSpaceAttr);

    auto wt_data_vals = generateWeights(wt_data_shape, weightsType, builder.getContext(), weight_file_name);
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }

    auto weight_data_ddr = funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightData_ddr_type,
                                                                wt_data_attr.reorder(DimsOrder::NHWC));

    // weights cmx tensor
    auto wtData_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtData_cmx_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps,
                                                 wtData_cmx_memSpaceAttr);
    auto wtData_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtData_cmx_type,
                                                                 VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                 /*data idx=*/WEIGHTS_CMX_OFFSET);

    // input - output cmx tensors
    auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto inputcmx_type =
            mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, inputcmx_memSpaceAttr);
    auto inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                               VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto outputcmx_type =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, outputcmx_memSpaceAttr);
    auto outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMAs
    /* auto in_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), funcinput, inputcmx.getOperation()->getResult(0), mlir::ValueRange(),
            mlir::ValueRange(barrier0.barrier()), false);
    /* auto wt_data_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), weight_data_ddr.getOperation()->getResult(0),
            wtData_cmx.getOperation()->getResult(0), mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);

    // Activation Window ddr
    SmallVector<int64_t> sparsity_shape;
    mlir::Type sparsity_type = getUInt8Type(builder.getContext());
    bool isOutFloat = false;
    if (outputType.isBF16() || outputType.isF16()) {
        isOutFloat = true;
    }
    mlir::IntegerAttr actChannelLength;
    auto sparsityAttr = getactivationWindow(builder, filter_size, stried_vec, out_shape[1], in_shape[1], sparsity_type,
                                            sparsity_shape, actChannelLength, isOutFloat, false, true);

    auto activationWindow_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    auto activationWindowAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(sparsity_shape));
    auto activationWindow_ddr_type = mlir::MemRefType::get(
            makeArrayRef(sparsity_shape), sparsity_type, activationWindowAffineMaps, activationWindow_ddr_memSpaceAttr);
    auto activationWindow_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), activationWindow_ddr_type,
                                                 Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));

    auto activationwindow_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto activationwindow_totalsize_bytes = activationwindow_totalsize * sparsity_type.getIntOrFloatBitWidth() / 8;

    // Activation Window cmx
    auto actWindow_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto actWindow_cmx_type = mlir::MemRefType::get(makeArrayRef(sparsity_shape), sparsity_type,
                                                    activationWindowAffineMaps, actWindow_cmx_memSpaceAttr);
    auto actWindow_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), actWindow_cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
            /*data idx=*/ACTIVATIONWINDOW_CMX_OFFSET);

    // activation window dma ddr->cmx
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), activationWindow_ddr.getOperation()->getResult(0),
                                       actWindow_cmx.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(barrier0.barrier()), false);

    // weights table ddr tensor
    SmallVector<int64_t> wtTbl_data_shape{sparsity_shape[0], 1, 1, 4};
    auto weightTbl_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    auto weightTblAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wtTbl_data_shape));
    auto weightTblData_ddr_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                  weightTblAffineMaps, weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    const std::vector<int32_t> wtTbl_data_values_vec = generateWeightsTablesValuesWithSparsity(
            testDesc, inputcmx_type, outputcmx_type, wtData_cmx_type, actWindow_cmx_type, ACTIVATIONWINDOW_CMX_OFFSET,
            wtTbl_data_shape, WEIGHTS_CMX_OFFSET);

    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor

    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;
    auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtTbl_cmx_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                  weightTblAffineMaps, wtTbl_cmx_memSpaceAttr);
    auto wtTbl_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtTbl_cmx_type,
                                                                VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                /*data idx=*/WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    /* auto wtTbl_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), weightTbl_data_ddr.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stried_vec);
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
            wtData_cmx.getOperation()->getResult(0), wtTbl_cmx.getOperation()->getResult(0),
            actWindow_cmx.getOperation()->getResult(0), parent_inputcmx.getOperation()->getResult(0),
            parent_outputcmx.getOperation()->getResult(0), outputcmx.getOperation()->getResult(0),
            mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::DWCONV,
            filtersize, strides, kernel_padding, actChannelLength);

    nceTask.addPPETask(funcbuilder);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int32_t> end_vec{static_cast<int32_t>(out_shape[3] - 1), static_cast<int32_t>(out_shape[2] - 1),
                                 static_cast<int32_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, padding_vec[PAD_NCETASK_LEFT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_RIGHT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_TOP]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_BOTTOM]), builder.getContext());

    /* auto dpuTask = */ variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), nullptr, start, end, pad,
                                                                 VPUIP::MPEMode::CUBOID_16x16);

    /* auto cmx_out_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput,
            mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(), false);

    // TODO : return empty as func does not return anything
    /* auto returnOp = */ funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC)});
}

}  // namespace hwtest
}  // namespace vpux
