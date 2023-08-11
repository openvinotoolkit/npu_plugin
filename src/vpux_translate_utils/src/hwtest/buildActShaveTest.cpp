//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

void buildActShave(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, const SmallVector<mlir::Type>& inputTypes, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    //  Input/Output -----------------------------------------------------------
    auto inputList = testDesc.getInputLayerList();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<SmallVector<int64_t>> inShapes;
    SmallVector<mlir::Type> funcInputTypes;

    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        SmallVector<int64_t> inShape(inputList[idx].shape.begin(), inputList[idx].shape.end());
        VPUX_THROW_UNLESS(!inShape.empty(), "buildActShave: Got empty input '{0}' shape ", idx);
        inShapes.push_back(inShape);

        auto inputParamType =
                getMemRefType(VPURT::BufferSection::NetworkInput, inShapes[idx], inputTypes[idx], DimsOrder::NHWC);
        funcInputTypes.push_back(inputParamType);
    }

    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());
    VPUX_THROW_UNLESS(!outShape.empty(), "buildActShave: Got empty outputShape");

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    funcInputTypes.push_back(outputParamType);

    // Build Function ---------------------------------------------------------------

    const auto funcType = builder.getFunctionType(makeArrayRef(funcInputTypes), outputParamType);

    std::string funcOpName = "actshave_";
    for (auto iType : funcInputTypes) {
        funcOpName += printToString("{0}", iType);
    }
    funcOpName += printToString("{0}", outputType);

    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcOpName, funcType,
                                                   builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    SmallVector<mlir::Value> funcinputs;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        auto funcinput = func.getArgument(idx);
        funcinputs.push_back(funcinput);
    }
    auto funcoutput = func.getArgument(inputList.size());

    //  Build main function: barriers
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    //  Build main function: input/output cmx
    SmallVector<vpux::VPURT::DeclareBufferOp> inputcmxVec;
    auto inputcmx_offset = 0;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        auto inputcmx_type =
                getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShapes[idx], inputTypes[idx], DimsOrder::NHWC);
        inputcmxVec.push_back(
                createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, inputcmx_offset));
        inputcmx_offset += totalTensorSize(inShapes[idx], inputTypes[idx]);
    }

    const auto outputcmx_offset = inputcmx_offset;
    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    vpux::VPURT::DeclareBufferOp outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, outputcmx_offset);

    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        //  Build main function: DMA func input -> CMX input
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(),
                                                    mlir::ValueRange(barrier0.barrier()), builder.getUnknownLoc(),
                                                    funcinputs[idx], getTensorResult(inputcmxVec[idx]));
    }

    //  Build main function: Call operation builder
    buildActShaveTask(testDesc, module, funcbuilder, log, makeArrayRef(funcInputTypes), inputcmxVec, outputcmx,
                      mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()));

    //  Build main function: DMA CMX output -> func output
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                builder.getUnknownLoc(), getTensorResult(outputcmx), funcoutput);

    //  Build main function: returnOp
    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    //  Pass Manager
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::ReferenceHW, 1, 1, None,
                                           log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    SmallVector<mlir::Type> inputTensorTypeVec;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        auto inputTensorType = getTensorType(ShapeRef(inShapes[idx]), inputTypes[idx], DimsOrder::NHWC, nullptr);
        inputTensorTypeVec.push_back(inputTensorType);
    }
    auto outputTensorType = getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr);

    //  CNN Operation
    buildCNNOp(builder, func.getName(), inputTensorTypeVec, outputTensorType);
}  // namespace hwtest

}  // namespace hwtest
}  // namespace vpux
