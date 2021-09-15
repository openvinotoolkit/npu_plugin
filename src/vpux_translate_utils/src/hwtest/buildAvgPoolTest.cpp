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

#include <climits>

namespace vpux {
namespace hwtest {

void buildAvgpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
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

    auto output_totalsize = totalTensorSize(out_shape, outputType);

    SmallVector<int64_t> wt_data_shape{in_shape[1], 1, pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};

    auto scaleValue = 1 / double(pool_op.kernel_shape.at(0) * pool_op.kernel_shape.at(1));

    mlir::Type weightsType = inputType;

    if (auto qtype = inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        auto inputStorageType = mlir::quant::QuantizedType::castToStorageType(qtype);
        int64_t zeroPoint = 0;

        if (inputStorageType.isUnsignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else if (inputStorageType.isSignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 getSInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else {
            VPUX_THROW("Unsupported storage type for input quantized type. I8 or U8 is supported only");
        }
    }

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;

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

    auto func = builder.create<mlir::FuncOp>(loc, llvm::formatv("avgPool_{0}_{1}", inputType, outputType).str(),
                                             funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    const auto weightDataAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(wt_data_shape));

    // weights tensorref
    auto wtData_memSpaceAttr = VPUIP::MemoryLocationAttr::get(ctx, VPUIP::MemoryLocation::AbsoluteAddr);
    auto wtData_type =
            mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps, wtData_memSpaceAttr);
    auto wtData = funcbuilder.create<VPUIP::DeclareTensorOp>(loc, wtData_type, VPUIP::MemoryLocation::AbsoluteAddr,
                                                             /*locale ndex=*/0,
                                                             /*data idx=*/0);

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

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            loc, outputcmx_type, inputcmx.getOperation()->getResult(0), wtData.getOperation()->getResult(0),
            mlir::Value(), mlir::Value(), parent_inputcmx.getOperation()->getResult(0),
            parent_outputcmx.getOperation()->getResult(0), outputcmx.getOperation()->getResult(0),
            mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::AVEPOOL,
            filtersize, strides, kernel_padding, mlir::IntegerAttr(), nullptr);

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
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
