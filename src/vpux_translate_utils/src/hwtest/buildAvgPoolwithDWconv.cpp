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

#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

#include <climits>

namespace vpux {
namespace hwtest {

void buildAvgpoolWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayer();
    auto pool_op = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    std::vector<int64_t> filter_size{pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};
    std::vector<int64_t> stride_vec(pool_op.stride.begin(), pool_op.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(pool_op.pad);

    auto input_totalsize = totalTensorSize(in_shape, inputType);
    auto output_totalsize = totalTensorSize(out_shape, outputType);

    SmallVector<int64_t> wt_data_shape{in_shape[1], 1, pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};

    auto scaleValue = 1 / double(pool_op.kernel_shape.at(0) * pool_op.kernel_shape.at(1));

    mlir::Type weightsType = inputType;

    if (auto qtype = inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        weightsType = mlir::quant::QuantizedType::castToStorageType(qtype);
        int64_t zeroPoint = 0;
        if (weightsType.isSignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 getSInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else {
            weightsType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        }
    }

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTS_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;

    SmallVector<mlir::Type> inputTypes;
    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(in_shape));
    inputTypes.push_back(
            getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableInput, in_shape, inputType, inputAffineMaps));
    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(out_shape));
    auto outputParamType =
            getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableOutput, out_shape, outputType, outputAffineMaps);
    inputTypes.push_back(outputParamType);
    SmallVector<ArrayRef<mlir::AffineMap>> argsAffineMaps{inputAffineMaps, outputAffineMaps};

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(loc, llvm::formatv("avgPool_{0}_{1}", inputType, outputType).str(),
                                             funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data

    // Generate weights for kh x kw DW conv

    const auto weightDataffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(wt_data_shape));
    auto weightData_ddr_type2 = getMemRefType(funcbuilder, VPUIP::MemoryLocation::GraphFile, wt_data_shape, weightsType,
                                              weightDataffineMaps);
    size_t weightDataSize = static_cast<size_t>(std::accumulate(wt_data_shape.begin(), wt_data_shape.end(),
                                                                static_cast<int64_t>(1), std::multiplies<int64_t>()));

    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_data_shape, weightsType);
    if (auto qtype = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_data_shape, getSInt8Type(ctx));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_data_shape, getUInt8Type(ctx));
        }
    }
    mlir::DenseElementsAttr wt_data_valss;
    if (weightsType.isF16()) {
        std::vector<float16> wt_vec(weightDataSize, static_cast<float>(scaleValue));
        wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<float16>(wt_vec));
    } else if (weightsType.isBF16()) {
        std::vector<bfloat16> wt_vec(weightDataSize, static_cast<float>(scaleValue));
        wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<bfloat16>(wt_vec));
    } else {
        scaleValue = 1;
        if (weightsType.dyn_cast<mlir::quant::QuantizedType>().getFlags() & mlir::quant::QuantizationFlags::Signed) {
            std::vector<int8_t> wt_vec(weightDataSize, static_cast<int8_t>(scaleValue));
            wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<int8_t>(wt_vec));
        } else {
            std::vector<uint8_t> wt_vec(weightDataSize, static_cast<uint8_t>(scaleValue));
            wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<uint8_t>(wt_vec));
        }
    }
    auto wt_data_attr = Const::ContentAttr::get(wt_data_valss);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto weight =
            funcbuilder.create<Const::DeclareOp>(loc, weightData_ddr_type2, wt_data_attr.reorder(DimsOrder::NHWC));

    auto weight_data_ddr = VPUIP::alignDepthwiseWeightTensor(funcbuilder, loc, weight.getResult());

    auto wt_data_shape_padded = weight_data_ddr.getType().cast<mlir::ShapedType>().getShape();
    const auto weightDataPaddedAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(wt_data_shape_padded));

    // weights cmx tensor
    auto wtData_cmx_type = getMemRefType(funcbuilder, VPUIP::MemoryLocation::VPU_CMX_NN,
                                         to_vector<4>(wt_data_shape_padded), weightsType, weightDataPaddedAffineMaps);
    auto wtData_cmx = createDeclareTensorOp(funcbuilder, wtData_cmx_type, 0, WEIGHTS_CMX_OFFSET);

    auto weight_padded_totalsize = totalTensorSize(wt_data_shape_padded, weightsType);
    const auto ACTIVATIONWINDOW_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weight_padded_totalsize;

    // input - output cmx tensors
    auto inputcmx_type =
            getMemRefType(funcbuilder, VPUIP::MemoryLocation::VPU_CMX_NN, in_shape, inputType, inputAffineMaps);
    auto inputcmx = createDeclareTensorOp(funcbuilder, inputcmx_type, 0, INPUT_CMX_OFFSET);

    auto outputcmx_type =
            getMemRefType(funcbuilder, VPUIP::MemoryLocation::VPU_CMX_NN, out_shape, outputType, outputAffineMaps);
    auto outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx = createDeclareTensorOp(funcbuilder, inputcmx_type, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(loc, 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(loc, 1);

    // DMAs
    funcbuilder.create<VPUIP::NNDMAOp>(loc, funcinput, inputcmx.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(barrier0.barrier()), false);
    funcbuilder.create<VPUIP::NNDMAOp>(loc, weight_data_ddr, wtData_cmx.getOperation()->getResult(0),
                                       mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);

    const auto bitPatternSize = VPUIP::NCESparsity::getBitPatternSize(
            makeArrayRef(filter_size), stride_vec[0],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType);
    mlir::IntegerAttr actChannelLength = funcbuilder.getI32IntegerAttr(checked_cast<int32_t>(bitPatternSize));

    const auto fakeSparsity = VPUIP::NCESparsity::getFakeSparsity(
            makeArrayRef(filter_size), stride_vec[0],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            in_shape[1]);

    const auto sparsity_type = getUInt8Type(ctx);
    int64_t numChannels = in_shape[1];
    SmallVector<int64_t> sparsity_shape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    const auto dataStorageType = mlir::RankedTensorType::get(sparsity_shape, sparsity_type);
    const auto sparsityAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(fakeSparsity));

    auto activationWindowAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(sparsity_shape));
    auto activationWindow_ddr_type = getMemRefType(funcbuilder, VPUIP::MemoryLocation::GraphFile, sparsity_shape,
                                                   sparsity_type, activationWindowAffineMaps);
    auto activationWindow_ddr = funcbuilder.create<Const::DeclareOp>(
            loc, activationWindow_ddr_type, Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));

    auto activationwindow_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto activationwindow_totalsize_bytes =
            activationwindow_totalsize * sparsity_type.getIntOrFloatBitWidth() / CHAR_BIT;

    // Activation Window cmx
    auto actWindow_cmx_type = getMemRefType(funcbuilder, VPUIP::MemoryLocation::VPU_CMX_NN, sparsity_shape,
                                            sparsity_type, activationWindowAffineMaps);
    auto actWindow_cmx = createDeclareTensorOp(funcbuilder, actWindow_cmx_type, 0, ACTIVATIONWINDOW_CMX_OFFSET);

    // activation window dma ddr->cmx
    funcbuilder.create<VPUIP::NNDMAOp>(loc, activationWindow_ddr.getOperation()->getResult(0),
                                       actWindow_cmx.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(barrier0.barrier()), false);

    // weights table ddr tensor
    SmallVector<int64_t> wtTbl_data_shape{sparsity_shape[0], 1, 1, 4};
    auto weightTblAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(wtTbl_data_shape));
    auto weightTblData_ddr_type = getMemRefType(funcbuilder, VPUIP::MemoryLocation::GraphFile, wtTbl_data_shape,
                                                builder.getIntegerType(32, true), weightTblAffineMaps);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, true));

    auto weights_outChannel = wtData_cmx_type.getShape()[0];
    auto weights_set_size =
            wtData_cmx_type.getShape()[1] * wtData_cmx_type.getShape()[2] * wtData_cmx_type.getShape()[3];
    size_t elementsize_bytes = 0;
    if (auto qType = wtData_cmx_type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementsize_bytes = qType.getStorageType().getIntOrFloatBitWidth() / CHAR_BIT;

    } else {
        elementsize_bytes = (wtData_cmx_type.getElementType().getIntOrFloatBitWidth()) / CHAR_BIT;
    }
    auto weights_set_nbytes = weights_set_size * elementsize_bytes;

    const std::vector<int32_t> wtTbl_data_values_vec = vpux::VPUIP::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<int32_t>(WEIGHTS_CMX_OFFSET), static_cast<int32_t>(weights_set_nbytes),
            static_cast<int32_t>(ACTIVATIONWINDOW_CMX_OFFSET), vpux::VPUIP::ArchKind::MTL, weights_outChannel,
            weightsType);

    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr = funcbuilder.create<Const::DeclareOp>(
            loc, weightTblData_ddr_type, Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor

    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;
    auto wtTbl_cmx_type = getMemRefType(funcbuilder, VPUIP::MemoryLocation::VPU_CMX_NN, wtTbl_data_shape,
                                        builder.getIntegerType(32, true), weightTblAffineMaps);
    auto wtTbl_cmx = createDeclareTensorOp(funcbuilder, wtTbl_cmx_type, 0, WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    funcbuilder.create<VPUIP::NNDMAOp>(loc, weightTbl_data_ddr.getOperation()->getResult(0),
                                       wtTbl_cmx.getOperation()->getResult(0), mlir::ValueRange(),
                                       mlir::ValueRange(barrier0.barrier()), false);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            loc, outputcmx_type, inputcmx.getOperation()->getResult(0), wtData_cmx.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), actWindow_cmx.getOperation()->getResult(0),
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::DWCONV, filtersize, strides, kernel_padding,
            actChannelLength, nullptr);

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
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_BOTTOM]), ctx);

    variantbuilder.create<VPUIP::DPUTaskOp>(loc, start, end, pad, VPUIP::MPEMode::CUBOID_16x16);

    funcbuilder.create<VPUIP::NNDMAOp>(loc, outputcmx.getOperation()->getResult(0), funcoutput,
                                       mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(), false);

    funcbuilder.create<mlir::ReturnOp>(loc, funcoutput);

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC)});
}

}  // namespace hwtest
}  // namespace vpux
