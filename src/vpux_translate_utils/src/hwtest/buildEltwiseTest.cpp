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
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildEltwiseAdd(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayer();
    auto weight = testDesc.getWeightLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> weights_shape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto INPUT1_CMX_OFFSET = INPUT0_CMX_OFFSET + input_totalsize;

    VPUX_THROW_UNLESS((inputType == weightsType), "Eltwise expects inputs of same type");

    SmallVector<mlir::Type> inputTypes;
    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(in_shape));
    inputTypes.push_back(
            getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableInput, in_shape, inputType, inputAffineMaps));
    inputTypes.push_back(getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableInput, weights_shape, weightsType,
                                       inputAffineMaps));

    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(ctx, Shape(out_shape));
    auto outputParamType =
            getMemRefType(builder, VPUIP::MemoryLocation::ProgrammableOutput, out_shape, outputType, outputAffineMaps);
    inputTypes.push_back(outputParamType);
    SmallVector<ArrayRef<mlir::AffineMap>> argsAffineMaps{inputAffineMaps, inputAffineMaps, outputAffineMaps};

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("eltwise_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcweights = func.getArgument(1);
    auto funcoutput = func.getArgument(2);

    // input - output cmx tensors
    auto inputcmx_type =
            getMemRefType(builder, VPUIP::MemoryLocation::VPU_CMX_NN, in_shape, inputType, inputAffineMaps);
    auto inputcmx = createDeclareTensorOp(funcbuilder, inputcmx_type, 0, INPUT0_CMX_OFFSET);

    auto weightscmx_type =
            getMemRefType(builder, VPUIP::MemoryLocation::VPU_CMX_NN, weights_shape, weightsType, inputAffineMaps);
    auto weightscmx = createDeclareTensorOp(funcbuilder, weightscmx_type, 0, INPUT1_CMX_OFFSET);

    auto outputcmx_type =
            getMemRefType(builder, VPUIP::MemoryLocation::VPU_CMX_NN, out_shape, outputType, outputAffineMaps);
    auto outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx = createDeclareTensorOp(funcbuilder, inputcmx_type, 0, INPUT0_CMX_OFFSET);
    auto parent_outputcmx = createDeclareTensorOp(funcbuilder, outputcmx_type, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMAs
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcinput, inputcmx.getOperation()->getResult(0),
                                       mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), funcweights, weightscmx.getOperation()->getResult(0),
                                       mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);
    funcbuilder.create<VPUIP::NNDMAOp>(builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput,
                                       mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(), false);

    // NCE Task
    mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);
    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
            weightscmx.getOperation()->getResult(0), mlir::Value(), nullptr,
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::ELTWISE, mlir::ArrayAttr(), mlir::ArrayAttr(),
            mlir::ArrayAttr(), actChannelLength, /*is_continued*/ nullptr);

    nceTask.addPPETask(funcbuilder);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int32_t> end_vec{static_cast<int32_t>(out_shape[3] - 1), static_cast<int32_t>(out_shape[2] - 1),
                                 static_cast<int32_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, 0), getIntAttr(builder, 0), getIntAttr(builder, 0),
                                       getIntAttr(builder, 0), ctx);

    // NB For eltwise operations, NTHW_NTK=(8, 8) is the only mode supported by
    // the hardware; this corresponds to CUBOID_8x16.
    nceTask.addDPUTask(variantbuilder, start, end, pad, VPUIP::MPEMode::CUBOID_8x16);

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(),
               {getTensorType(in_shape, inputType, DimsOrder::NHWC),
                getTensorType(weights_shape, weightsType, DimsOrder::NHWC)},
               {getTensorType(in_shape, outputType, DimsOrder::NHWC)});
}
}  // namespace hwtest
}  // namespace vpux
