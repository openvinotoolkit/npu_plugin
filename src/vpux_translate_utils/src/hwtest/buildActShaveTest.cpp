//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/hwtests_ops.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

void buildActShave(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    //  Input/Output -----------------------------------------------------------

    auto input = testDesc.getInputLayer();
    auto output = testDesc.getOutputLayer();
    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildActShave: Got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildActShave: Got empty outputShape");

    SmallVector<mlir::Type> inputTypes;
    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    inputTypes.push_back(inputParamType);
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    // Build Function ---------------------------------------------------------------

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             llvm::formatv("actshave_{0}_{1}", inputType, outputType).str(), funcType,
                                             builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    //  Build main function: input/output cmx
    const auto inputcmx_offset = 0;
    const auto outputcmx_offset = inputcmx_offset + totalTensorSize(inShape, inputType);

    auto inputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, inShape, inputType, DimsOrder::NHWC);

    vpux::VPURT::DeclareBufferOp inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, inputcmx_offset);

    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, outShape, outputType, DimsOrder::NHWC);
    vpux::VPURT::DeclareBufferOp outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, outputcmx_offset);

    //  Build main function: barriers
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    //  Build main function: DMA func input -> CMX input
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                builder.getUnknownLoc(), funcinput, getTensorResult(inputcmx));

    //  Build main function: Call operation builder
    buildActShaveTask(testDesc, module, funcbuilder, log, inputTypes, inputcmx, outputcmx,
                      mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()));

    //  Build main function: DMA CMX output -> func output
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                builder.getUnknownLoc(), getTensorResult(outputcmx), funcoutput);

    //  Build main function: returnOp
    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    //  Pass Manager
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::ReferenceHW, 1, log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    //  CNN Operation
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
