//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

void buildMaxPool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayer();
    auto pool_op = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(in_shape.size() >= 4, "buildMaxPool: Input rank is less than 4");
    VPUX_THROW_UNLESS(out_shape.size() >= 4, "buildMaxPool: Output rank is less than 4");

    std::vector<int64_t> filter_size{pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};
    std::vector<int64_t> stride_vec(pool_op.stride.begin(), pool_op.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(pool_op.pad);

    auto input_totalsize = totalTensorSize(in_shape, inputType);
    auto output_totalsize = totalTensorSize(out_shape, outputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto ACTIVATIONWINDOW_CMX_OFFSET = INPUT0_CMX_OFFSET + input_totalsize;

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC));

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             printToString("maxpool_{0}_{1}", inputType, outputType), funcType,
                                             builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput0 = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // input - output cmx tensors
    auto input0cmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, in_shape, inputType, DimsOrder::NHWC);
    auto input0cmx =
            createDeclareTensorOp(funcbuilder, input0cmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);

    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, out_shape, outputType, DimsOrder::NHWC);
    auto outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_input0cmx =
            createDeclareTensorOp(funcbuilder, input0cmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);
    auto parent_outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMA input-->cmx
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), funcinput0, input0cmx.getOperation()->getResult(0));

    mlir::Type uderlyingInputType = inputType.isa<mlir::quant::QuantizedType>()
                                            ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                            : inputType;
    // Generate activation window

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, ShapeRef(filter_size),
                                                                    stride_vec[1], uderlyingInputType, in_shape[1]);
    mlir::IntegerAttr actChannelLength = funcbuilder.getI32IntegerAttr(checked_cast<int32_t>(bitPatternSize));

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::POOL, ShapeRef(filter_size),
                                                                stride_vec[1], uderlyingInputType, in_shape[1]);

    const auto elemType = getUInt8Type(ctx);
    SmallVector<int64_t> sparsity_shape{1, 1, 1, static_cast<int64_t>(fakeSparsity.size())};

    const auto dataStorageType = mlir::RankedTensorType::get(sparsity_shape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(fakeSparsity));

    const auto dataType = getMemRefType(VPURT::BufferSection::DDR, sparsity_shape, elemType, DimsOrder::NHWC);

    auto dataConstOp = funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), dataType,
                                                            Const::ContentAttr::get(dataAttr).reorder(DimsOrder::NHWC));

    auto activationwindow_totalsize = totalTensorSize(sparsity_shape, elemType);
    auto activationwindow_totalsize_bytes = activationwindow_totalsize * elemType.getIntOrFloatBitWidth() / CHAR_BIT;

    // Activation Window cmx
    auto actWindow_cmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, sparsity_shape, elemType, DimsOrder::NHWC);
    auto actWindow_cmx = createDeclareTensorOp(funcbuilder, actWindow_cmx_type, VPURT::BufferSection::CMX_NN, 0,
                                               ACTIVATIONWINDOW_CMX_OFFSET);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), dataConstOp.getOperation()->getResult(0),
                                          actWindow_cmx.getOperation()->getResult(0));
    // weights table ddr tensor
    SmallVector<int64_t> wtTbl_data_shape{output.shape[1], 1, 1, 4};
    auto weightTblData_ddr_type = getMemRefType(VPURT::BufferSection::DDR, wtTbl_data_shape,
                                                builder.getIntegerType(32, true), DimsOrder::NHWC);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    const std::vector<int32_t> wtTbl_data_values_vec =
            VPU::NCESparsity::getWeightsTable(inputType, outputType, static_cast<int32_t>(0), static_cast<int32_t>(0),
                                              static_cast<int32_t>(ACTIVATIONWINDOW_CMX_OFFSET),
                                              static_cast<int32_t>(0), testDesc.getArchitecture(), output.shape[1]);

    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor

    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;
    auto wtTbl_cmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, wtTbl_data_shape,
                                        builder.getIntegerType(32, true), DimsOrder::NHWC);
    auto wtTbl_cmx = createDeclareTensorOp(funcbuilder, wtTbl_cmx_type, VPURT::BufferSection::CMX_NN, 0,
                                           WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), weightTbl_data_ddr.getOperation()->getResult(0),
                                          wtTbl_cmx.getOperation()->getResult(0));

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);
    auto kernel_padding = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                              padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()),
            builder.getUnknownLoc(), outputcmx_type, input0cmx.getOperation()->getResult(0), mlir::Value(),
            wtTbl_cmx.getOperation()->getResult(0), actWindow_cmx.getOperation()->getResult(0),
            parent_input0cmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), VPUIP::NCETaskType::MAXPOOL, filtersize, strides, kernel_padding,
            actChannelLength, nullptr, /*sp_pattern*/ nullptr);

    nceTask.addPPETask(funcbuilder);

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int32_t> end_vec{static_cast<int32_t>(out_shape[3] - 1), static_cast<int32_t>(out_shape[2] - 1),
                                 static_cast<int32_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                   padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, end, pad, VPU::MPEMode::CUBOID_16x16);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                          builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput);

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);
    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
