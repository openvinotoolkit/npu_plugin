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

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildMaxpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto input = testDesc.getInputLayer();
    auto maxpool = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    std::vector<int64_t> filter_size{maxpool.kernel_shape.at(0), maxpool.kernel_shape.at(1)};
    std::vector<int64_t> stried_vec(maxpool.stride.begin(), maxpool.stride.end());
    auto padding_vec = convertNBPadtoNCETaskPad(maxpool.pad);

    auto totaltensorsize = [&](SmallVector<int64_t>& shape, mlir::Type elementtype) {
        size_t numbytes = 0;
        if (auto qType = elementtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            numbytes = qType.getStorageType().getIntOrFloatBitWidth() / 8;
        } else {
            numbytes = elementtype.getIntOrFloatBitWidth() / 8;
        }
        size_t totalsize = static_cast<size_t>(
                std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

        return (totalsize * numbytes);
    };

    auto output_totalsize = totaltensorsize(out_shape, outputType);
    auto input_totalsize = totaltensorsize(in_shape, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto ACTIVATIONWINDOW_CMX_OFFSET = INPUT0_CMX_OFFSET + input_totalsize;

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
    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             llvm::formatv("maxpool_{0}_{1}", inputType, outputType).str(), funcType,
                                             builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput0 = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // input - output cmx tensors
    auto input0cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto input0cmx_type = getMemRefType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, input0cmx_memSpaceAttr);
    auto input0cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), input0cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT0_CMX_OFFSET);

    auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto outputcmx_type = getMemRefType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, outputcmx_memSpaceAttr);
    auto outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_input0cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), input0cmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT0_CMX_OFFSET);
    auto parent_outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMA input-->cmx
    /* auto in0_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), funcinput0, input0cmx.getOperation()->getResult(0), mlir::ValueRange(),
            mlir::ValueRange(barrier0.barrier()), false);

    // Activation Window ddr
    SmallVector<int64_t> sparsity_shape;
    mlir::Type sparsity_type = getUInt8Type(builder.getContext());
    bool isOutFloat = false;
    if (outputType.isBF16() || outputType.isF16()) {
        isOutFloat = true;
    }
    mlir::IntegerAttr actChannelLength;
    auto sparsityAttr = getactivationWindow(builder, filter_size, stried_vec, out_shape[1], in_shape[1], sparsity_type,
                                            sparsity_shape, actChannelLength, isOutFloat, true, false);

    auto activationWindow_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    auto activationWindow_ddr_type =
            getMemRefType(ShapeRef(sparsity_shape), sparsity_type, DimsOrder::NHWC, activationWindow_ddr_memSpaceAttr);
    auto activationWindow_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), activationWindow_ddr_type,
                                                 Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));
    auto activationwindow_totalsize = totaltensorsize(sparsity_shape, sparsity_type);
    auto activationwindow_totalsize_bytes = activationwindow_totalsize * sparsity_type.getIntOrFloatBitWidth() / 8;

    // Activation Window cmx
    auto actWindow_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto actWindow_cmx_type =
            getMemRefType(ShapeRef(sparsity_shape), sparsity_type, DimsOrder::NHWC, actWindow_cmx_memSpaceAttr);
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
    auto weightTblData_ddr_type =
            getMemRefType(ShapeRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC,
                          weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    const std::vector<int32_t> wtTbl_data_values_vec =
            generateWeightsTablesValuesForMaxPool(testDesc, input0cmx_type, outputcmx_type, actWindow_cmx_type,
                                                  ACTIVATIONWINDOW_CMX_OFFSET, wtTbl_data_shape);

    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor

    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;
    auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtTbl_cmx_type = getMemRefType(ShapeRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                        DimsOrder::NHWC, wtTbl_cmx_memSpaceAttr);
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
            builder.getUnknownLoc(), outputcmx_type, input0cmx.getOperation()->getResult(0), mlir::Value(),
            wtTbl_cmx.getOperation()->getResult(0), actWindow_cmx.getOperation()->getResult(0),
            parent_input0cmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::MAXPOOL, filtersize, strides, kernel_padding,
            actChannelLength, /*is_continued*/ nullptr, /*odu_permutation*/ nullptr);

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
    pm.addPass(createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode::ReferenceSW, None, log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
