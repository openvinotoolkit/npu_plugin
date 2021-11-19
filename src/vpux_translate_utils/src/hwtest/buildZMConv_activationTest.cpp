//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <limits>
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

void buildSimpleZMajorConvActivation(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                     mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                     mlir::Type outputType) {
    auto input = testDesc.getInputLayer();
    auto weight = testDesc.getWeightLayer();
    auto conv = testDesc.getConvLayer();
    auto activation = testDesc.getActivationLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    // SmallVector<int64_t> wt_data_shape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> wt_data_shape{weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]};

    SmallVector<int64_t> wtTbl_data_shape{wt_data_shape[0], 1, 1, 4};
    const char* weight_file_name = "weights.dat";

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);
    auto weightsTable_totalsize = /*always 4 bytes*/ 4 * wtTbl_data_shape[0] * wtTbl_data_shape[3];

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;
    const auto WEIGHTS_CMX_OFFSET = WEIGHTSTABLE_CMX_OFFSET + weightsTable_totalsize;

    SmallVector<mlir::Type> inputTypes;
    auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    inputTypes.push_back(getMemRefType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, memSpaceAttr_in));
    auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    auto outputParamType = getMemRefType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, memSpaceAttr_out);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("zmajor_conv_{0}_{1}_{2}_{3}", to_string(activation.activationType), inputType, weightsType,
                          outputType)
                    .str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data
    auto weight_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    auto weightData_ddr_type =
            getMemRefType(ShapeRef(wt_data_shape), weightsType, DimsOrder::NHWC, weight_data_ddr_memSpaceAttr);

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
    auto wtData_cmx_type =
            getMemRefType(ShapeRef(wt_data_shape), weightsType, DimsOrder::NHWC, wtData_cmx_memSpaceAttr);
    auto wtData_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtData_cmx_type,
                                                                 VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                 /*data idx=*/WEIGHTS_CMX_OFFSET);

    // input - output cmx tensors
    auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto inputcmx_type = getMemRefType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, inputcmx_memSpaceAttr);
    auto inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                               VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto outputcmx_type = getMemRefType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, outputcmx_memSpaceAttr);
    auto outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // weights table ddr tensor
    auto weightTbl_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    auto weightTblData_ddr_type =
            getMemRefType(ShapeRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC,
                          weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    const std::vector<int32_t> wtTbl_data_values_vec =
            generateWeightsTablesValues(testDesc, WEIGHTS_CMX_OFFSET, inputcmx_type, outputcmx_type, wtData_cmx_type);
    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtTbl_cmx_type = getMemRefType(ShapeRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                        DimsOrder::NHWC, wtTbl_cmx_memSpaceAttr);
    auto wtTbl_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtTbl_cmx_type,
                                                                VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                /*data idx=*/WEIGHTSTABLE_CMX_OFFSET);

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
    /* auto wtTbl_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), weightTbl_data_ddr.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);
    /* auto cmx_out_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput,
            mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(), false);

    // NCE Task
    auto strides = getIntArrayAttr(builder, conv.stride);
    std::vector<int64_t> padding_vec{conv.pad[0], conv.pad[2], conv.pad[1], conv.pad[3]};
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);
    SmallVector<int64_t> kernel_vec = {wt_data_shape[2], wt_data_shape[3]};
    auto kernel_size = getIntArrayAttr(builder, kernel_vec);
    mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);
    auto ppeLayer = getPPELayerFromConfig(activation);
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t lreluMult = 1;
    uint32_t lreluShift = 0;

    calculateppeParams(testDesc, clampLow, clampHigh, lreluMult, lreluShift);

    auto instructionList = mlir::Value();
    if (activation.activationType == nb::ActivationType::LeakyReLU) {
        log.info("Generating instruction table for ", to_string(activation.activationType));
        // instructionList ddr tensor
        auto instructionList_ddr_memSpaceAttr =
                VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
        SmallVector<mlir::AffineMap> instructionList_ddr_affineMaps;
        std::size_t numberOfInstructions = 25;
        std::size_t alignedInstructions = round_up(numberOfInstructions, 16);
        llvm::SmallVector<int64_t> instructionList_data_shape = {1, 1, 1, static_cast<int64_t>(alignedInstructions)};
        auto instructionList_ddr_type =
                getMemRefType(ShapeRef(instructionList_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                              DimsOrder::NHWC, instructionList_ddr_memSpaceAttr);
        /* const auto instructionList_ddr_valueType = */
        mlir::RankedTensorType::get(instructionList_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

        const std::vector<int32_t> instructionList_values_vec =
                getInstructionListVals(activation.activationType, instructionList_data_shape);
        auto instructionList_data_values = makeArrayRef<int32_t>(instructionList_values_vec);
        auto instructionList_vals = mlir::DenseElementsAttr::get(instructionList_ddr_type, instructionList_data_values);
        /* auto instructionList_data_ddr = */ funcbuilder.create<Const::DeclareOp>(
                builder.getUnknownLoc(), instructionList_ddr_type,
                Const::ContentAttr::get(instructionList_vals).reorder(DimsOrder::NHWC));

        auto weights_totalsize = totalTensorSize(wt_data_shape, weightsType);
        const auto INSTRUCTIONLIST_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weights_totalsize;

        // instructionList cmx tensor
        auto instructionList_cmx_memSpaceAttr =
                VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
        auto instructionList_cmx_type =
                getMemRefType(ShapeRef(instructionList_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                              DimsOrder::NHWC, instructionList_cmx_memSpaceAttr);
        auto instructionList_cmx =
                funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), instructionList_cmx_type,
                                                           VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                           /*data idx=*/INSTRUCTIONLIST_CMX_OFFSET);
        instructionList = instructionList_cmx.getOperation()->getResult(0);
    }

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
            wtData_cmx.getOperation()->getResult(0), wtTbl_cmx.getOperation()->getResult(0), nullptr,
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::CONV, kernel_size, strides, kernel_padding,
            actChannelLength, /*is_continued*/ nullptr, /*odu_permutation*/ nullptr);

    nceTask.addPPETask(funcbuilder, ppeLayer, instructionList, clampLow, clampHigh, lreluMult, lreluShift);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int64_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int64_t> end_vec{static_cast<int64_t>(out_shape[3] - 1), static_cast<int64_t>(out_shape[2] - 1),
                                 static_cast<int64_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, conv.pad[0]), getIntAttr(builder, conv.pad[1]),
                                       getIntAttr(builder, conv.pad[2]), getIntAttr(builder, conv.pad[3]),
                                       builder.getContext());

    /* auto dpuTask = */ variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), nullptr, start, end, pad,
                                                                 VPUIP::MPEMode::CUBOID_16x16);

    // TODO : return empty as func does not return anything
    /* auto returnOp = */ funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
