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

#include "hwtest.hpp"

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace {

mlir::DenseElementsAttr generateWeights(SmallVector<int64_t>& wt_shape, mlir::Type dtype, mlir::MLIRContext* ctx) {
    mlir::DenseElementsAttr wt_data_vals;
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape, dtype);
    size_t vecSize = static_cast<size_t>(
            std::accumulate(wt_shape.begin(), wt_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        dtype = mlir::quant::QuantizedType::castToStorageType(qtype);
        // ConstContentBase uses raw u8 (char) type for data elements
        wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape, getUInt8Type(ctx));
    }

    if (dtype.isSignedInteger(4)) {
        const std::vector<int8_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<int8_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else if (dtype.isInteger(4)) {
        const std::vector<uint8_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<uint8_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else if (dtype.isSignedInteger(8)) {
        const std::vector<int8_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<int8_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else if (dtype.isInteger(8)) {
        const std::vector<uint8_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<uint8_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else if (dtype.isF16()) {
        const std::vector<float_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<float_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else if (dtype.isF32()) {
        const std::vector<float_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<float_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else if (dtype.isBF16()) {
        const std::vector<float_t> wt_vec(vecSize, 1);
        auto wt_data_values = makeArrayRef<float_t>(wt_vec);
        wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
        return wt_data_vals;
    } else {
        throw std::domain_error{llvm::formatv("Expected a valid weight data type; got {0}", dtype).str()};
    }
}

void computeQuantMultShift(float scale, unsigned& shift, unsigned& mult) {
    auto bits = 15;
    int exponent;
    double mantissa = std::frexp(scale, &exponent);
    shift = bits - exponent;
    mult = static_cast<unsigned>((mantissa * pow(2, bits)));
}

// Based on MCM logic to generate mult-shift element in weight table
// weights_table[outChannel][2] = mult << 16 | round << 14 | shift << 8 | prelu
size_t calcWeightsTableMultShift(mlir::MemRefType input, mlir::MemRefType output, mlir::MemRefType weights) {
    size_t multshift = 0;
    // 8bit mult mask
    static const uint32_t PRELU_MULT_MASK = 0x000000FF;
    // 6bit shift mask
    static const uint32_t PRELU_SHIFT_MASK = 0x00003F00;
    static const uint32_t PRELU_SHIFT_SHIFT = 8;
    // round mode mask
    static const uint32_t ROUND_MODE_MASK = 0x0000C000;
    static const uint32_t ROUND_MODE_SHIFT = 14;
    // scale mask
    static const uint32_t SCALE_MODE_MASK = 0xFFFF0000;
    static const uint32_t SCALE_MODE_SHIFT = 16;

    float out_scale = 1.0;
    float in_Scale = 1.0;
    float weights_Scale = 1.0;
    auto inputtype = input.getElementType();
    auto outtype = output.getElementType();
    auto wt_type = weights.getElementType();

    if (auto in_qType = inputtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        in_Scale = static_cast<float>(in_qType.getScale());
    }

    if (auto out_qType = outtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        out_scale = static_cast<float>(out_qType.getScale());
    }

    if (auto wt_qType = wt_type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        weights_Scale = static_cast<float>(wt_qType.getScale());
    }

    float result_Scale = (in_Scale * weights_Scale) / out_scale;

    auto float_as_int = [&](float f) {
        union bit_field32 {
            float fp;
            unsigned int ui;
        };
        bit_field32 v;
        v.fp = f;
        return v.ui;
    };

    if (input.getElementType().isBF16() || input.getElementType().isF16()) {
        multshift = float_as_int(result_Scale);
    } else {
        // harcoded
        int32_t round32 = 1;
        int32_t reluMult = 0;
        unsigned mult;
        unsigned shift;
        computeQuantMultShift(result_Scale, shift, mult);
        multshift = static_cast<int64_t>(
                ((mult << SCALE_MODE_SHIFT) & SCALE_MODE_MASK) | ((round32 << ROUND_MODE_SHIFT) & ROUND_MODE_MASK) |
                ((shift << PRELU_SHIFT_SHIFT) & PRELU_SHIFT_MASK) | (reluMult & PRELU_MULT_MASK));
    }

    return multshift;
}
std::vector<int32_t> generateWeightsTablesValues(size_t weights_offset, mlir::MemRefType input, mlir::MemRefType output,
                                                 mlir::MemRefType weights) {
    /*
    Each Weight table[4] represents per channel layout of weights data,
    For ex : {8448, 16777215, 1354584320, 0}
    with each index as :
    [0] -> Starting address of each weight data,
    [1] -> Sparsity map address,
    [2] -> mult << 16 | round << 14 | shift << 8 | prelu
    [3] -> bias ]
    */
    const size_t DATA_POINTER_IDX = 0;
    const size_t SPARSITY_POINTER_IDX = 1;
    const size_t MULTSHIFT_IDX = 2;
    const size_t BIAS_IDX = 3;
    const size_t BYTE_SIZE = 8;  // bits

    auto first_channel_offset = weights_offset;
    auto weights_outChannel = weights.getShape()[0];
    auto weights_set_size = weights.getShape()[1] * weights.getShape()[2] * weights.getShape()[3];
    size_t elementsize_bytes = 0;
    if (auto qType = weights.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementsize_bytes = qType.getStorageType().getIntOrFloatBitWidth() / BYTE_SIZE;

    } else {
        elementsize_bytes = (weights.getElementType().getIntOrFloatBitWidth()) / BYTE_SIZE;
    }
    auto weights_set_nbytes = weights_set_size * elementsize_bytes;

    // TODO: generic dtype
    int32_t bias_value = 0;

    // TODO: calculate
    // currently hard coded
    size_t sparsity_pointer = 16777215;

    auto mult_shift = calcWeightsTableMultShift(input, output, weights);

    // generate data pointers
    std::vector<int32_t> weightsTableVals(weights_outChannel * 4, 0);

    for (int64_t i = 0; i < weights_outChannel; ++i) {
        weightsTableVals[i * 4 + DATA_POINTER_IDX] = static_cast<int32_t>(first_channel_offset);
        first_channel_offset += weights_set_nbytes;

        weightsTableVals[i * 4 + SPARSITY_POINTER_IDX] = static_cast<int32_t>(sparsity_pointer);
        weightsTableVals[i * 4 + MULTSHIFT_IDX] = static_cast<int32_t>(mult_shift);
        weightsTableVals[i * 4 + BIAS_IDX] = bias_value;
    }

    return weightsTableVals;
}

void buildSimpleZMajorConv(mlir::ModuleOp module, mlir::OpBuilder builder, Logger& log, mlir::Type inputType,
                           mlir::Type weightsType, mlir::Type outputType) {
    const size_t num_inputs = 1;
    const size_t num_outputs = 1;
    const size_t num_func_args = 2;

    SmallVector<int64_t> in_shape{1, 16, 16, 16};
    SmallVector<int64_t> out_shape{1, 16, 16, 16};
    SmallVector<int64_t> wt_data_shape{16, 1, 1, 16};
    SmallVector<int64_t> wtTbl_data_shape{wt_data_shape[0], 1, 1, 4};

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
    auto weightsTable_totalsize = /*always 4 bytes*/ 4 * wtTbl_data_shape[0] * wtTbl_data_shape[3];

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;
    const auto WEIGHTS_CMX_OFFSET = WEIGHTSTABLE_CMX_OFFSET + weightsTable_totalsize;

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(num_func_args);
    SmallVector<mlir::AffineMap> in_affineMaps;
    in_affineMaps.push_back(vpux::DimsOrder::NHWC.toAffineMap(builder.getContext()));
    auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    inputTypes.push_back(mlir::MemRefType::get(makeArrayRef(in_shape), inputType, in_affineMaps, memSpaceAttr_in));
    auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    inputTypes.push_back(mlir::MemRefType::get(makeArrayRef(in_shape), outputType, in_affineMaps, memSpaceAttr_out));

    // TODO : pass outputTypes as empty as func does not return anything
    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(num_outputs);
    outputTypes.push_back(mlir::MemRefType::get(makeArrayRef(out_shape), outputType, in_affineMaps, memSpaceAttr_out));

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), makeArrayRef(outputTypes));

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("zmajor_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data
    auto weight_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    SmallVector<mlir::AffineMap> wtData_ddr_affineMaps;
    wtData_ddr_affineMaps.push_back(vpux::DimsOrder::NHWC.toAffineMap(builder.getContext()));
    auto weightData_ddr_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, wtData_ddr_affineMaps,
                                                     weight_data_ddr_memSpaceAttr);

    auto wt_data_vals = generateWeights(wt_data_shape, weightsType, builder.getContext());
    auto weight_data_ddr = funcbuilder.create<VPUIP::DeclareConstantTensorOp>(builder.getUnknownLoc(),
                                                                              weightData_ddr_type, wt_data_vals);

    // weights cmx tensor
    auto wtData_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtData_cmx_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, wtData_ddr_affineMaps,
                                                 wtData_cmx_memSpaceAttr);
    auto wtData_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtData_cmx_type,
                                                                 VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                 /*data idx=*/WEIGHTS_CMX_OFFSET);

    // input - output cmx tensors
    auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto inputcmx_type = mlir::MemRefType::get(makeArrayRef(in_shape), inputType, in_affineMaps, inputcmx_memSpaceAttr);
    auto inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                               VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto outputcmx_type =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, in_affineMaps, outputcmx_memSpaceAttr);
    auto outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // weights table ddr tensor
    auto weightTbl_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    SmallVector<mlir::AffineMap> wtTbl_ddr_affineMaps;
    wtTbl_ddr_affineMaps.push_back(vpux::DimsOrder::NHWC.toAffineMap(builder.getContext()));
    auto weightTblData_ddr_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                  wtTbl_ddr_affineMaps, weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    const std::vector<int32_t> wtTbl_data_values_vec =
            generateWeightsTablesValues(WEIGHTS_CMX_OFFSET, inputcmx_type, outputcmx_type, wtData_cmx_type);
    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr = funcbuilder.create<VPUIP::DeclareConstantTensorOp>(
            builder.getUnknownLoc(), weightTblData_ddr_type, wtTbl_data_vals);

    // weights table cmx tensor
    auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtTbl_cmx_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                  wtData_ddr_affineMaps, wtTbl_cmx_memSpaceAttr);
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
    std::vector<int32_t> stried_vec{1, 1};
    auto strides = getInt32ArrayAttr(builder.getContext(), stried_vec);
    std::vector<int32_t> padding_vec{0, 0, 0, 0};
    auto kernel_padding = getInt32ArrayAttr(builder.getContext(), padding_vec);

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
            wtData_cmx.getOperation()->getResult(0), wtTbl_cmx.getOperation()->getResult(0),
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::CONV, VPUIP::PPELayerTypeAttr(), kernel_padding,
            strides, 0);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getInt32ArrayAttr(builder.getContext(), start_vec);
    std::vector<int32_t> end_vec{15, 15, 15};
    auto end = getInt32ArrayAttr(builder.getContext(), end_vec);
    std::vector<int32_t> pad_begin_vec{0, 0};
    auto pad_begin = getInt32ArrayAttr(builder.getContext(), pad_begin_vec);
    std::vector<int32_t> pad_end_vec{0, 0};
    auto pad_end = getInt32ArrayAttr(builder.getContext(), pad_end_vec);

    /* auto dpuTask = */ variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, end, pad_begin,
                                                                 pad_end, VPUIP::MPEMode::CUBOID_16x16);

    // TODO : return empty as func does not return anything
    /* auto returnOp = */ funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(createSetCompileParamsPass(vpux::VPUIP::ArchKind::VPU3720, VPUIP::CompilationMode(), log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(builder.getContext(), func.getName());
    auto cnnOp = builder.create<IE::CNNNetworkOp>(builder.getUnknownLoc(), mainFuncName);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());

    for (size_t i = 0; i < num_inputs; ++i) {
        const auto& inputName = "input_" + std::to_string(i);
        const auto nameAttr = mlir::StringAttr::get(builder.getContext(), inputName);
        auto precision = inputType;
        if (auto qtype = precision.dyn_cast<mlir::quant::QuantizedType>()) {
            precision = mlir::quant::QuantizedType::castToStorageType(precision);
        }
        SmallVector<mlir::AffineMap> affineMaps;
        affineMaps.push_back(vpux::DimsOrder::NHWC.toAffineMap(builder.getContext()));
        auto memSpaceAttr_in =
                VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
        const auto userTypeAttr =
                mlir::TypeAttr::get(mlir::MemRefType::get(in_shape, precision, affineMaps, memSpaceAttr_in));
        inputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), builder.getListener());
    for (size_t i = 0; i < num_outputs; ++i) {
        const auto& resultName = "output_" + std::to_string(i);
        const auto nameAttr = mlir::StringAttr::get(builder.getContext(), resultName);
        auto precision = outputType;
        if (auto qtype = precision.dyn_cast<mlir::quant::QuantizedType>()) {
            precision = mlir::quant::QuantizedType::castToStorageType(precision);
        }
        SmallVector<mlir::AffineMap> affineMaps;
        affineMaps.push_back(vpux::DimsOrder::NHWC.toAffineMap(builder.getContext()));
        auto memSpaceAttr_out =
                VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
        const auto userTypeAttr =
                mlir::TypeAttr::get(mlir::MemRefType::get(out_shape, precision, affineMaps, memSpaceAttr_out));
        outputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr);
    }
}
}  // namespace

mlir::OwningModuleRef importHWTEST(llvm::StringRef, mlir::MLIRContext* ctx) {
    ctx->loadDialect<vpux::VPUIP::VPUIPDialect>();
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef("mainModule"));
    auto log = Logger{"vpux-hwtest", LogLevel::Info};
    auto builderLog = OpBuilderLogger{log.nest()};
    auto builder = mlir::OpBuilder(module.getBodyRegion());

    constexpr float scale = 1.f;
    constexpr size_t zero_p = 0;

    mlir::Type u8 =
            mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), builder.getF32Type(), scale, zero_p, 0, 255);

    // ZMajor convolutions
    //
    // TODO: Replace these hardcoded loops with a configuration input.
    //
    // We'd ideally like to write something like this, but we can only produce a
    // single module as our output, and the module can only contain a single
    // network.  And we want this to be driven from the JSON hardware test
    // description anyway.
    //
    // for (auto inputType : {i8, ui8}) {for (auto weightsType : {i8, ui8}) {for
    //     (auto outputType : {i8, ui8, f16}) {buildSimpleZMajorConv(builder,
    //     log, inputType, weightsType, outputType);
    //         }
    //     }
    // }
    // for (auto outputType : {i8, ui8, f16, f32})
    //     {buildSimpleZMajorConv(builder, log, f16, f16, outputType);
    // }
    // for (auto outputType : {bf16, f32}) {buildSimpleZMajorConv(builder, log,
    //     bf16, bf16, outputType);
    // }

    buildSimpleZMajorConv(module, builder, log, u8, u8, builder.getF16Type());

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    return module;
}

}  // namespace vpux
