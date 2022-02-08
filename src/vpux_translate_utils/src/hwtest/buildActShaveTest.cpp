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
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {
namespace {

IERT::KernelInfo getKernelInfo(nb::ActivationLayer activation, mlir::MLIRContext* ctx) {
    switch (activation.activationType) {
    case nb::ActivationType::HSwish:
        return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"hswish_fp16"}, {"hswish_fp16.cpp"}};
    case nb::ActivationType::Sigmoid:
        return IERT::KernelInfo{SmallVector<mlir::Attribute>{}, {"sigmoid_fp16"}, {"sigmoid_fp16.c"}};
    case nb::ActivationType::Softmax:
        return IERT::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, activation.axis)},
                                {"singleShaveSoftmax"},
                                {"single_shave_softmax.cpp"}};
    default:
        VPUX_THROW("Only HSwish, Sigmoid or Softmax activations is supported for ActShave tests");
    }
}

}  // namespace

void buildActShave(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto activation = testDesc.getActivationLayer();

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

    auto kernelInfo = getKernelInfo(activation, ctx);

    const auto convertToUnrankedType = [ctx](mlir::Type srcType) -> mlir::Type {
        auto type = srcType.dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), mlir::SymbolRefAttr::get(VPU::MemoryKindAttr::get(
                                                                            ctx, VPU::MemoryKind::CMX_NN)));
    };
    SmallVector<mlir::Type> inputTypesUnranked;
    std::transform(inputTypes.begin(), inputTypes.end(), std::back_inserter(inputTypesUnranked), convertToUnrankedType);
    std::transform(kernelInfo.args.begin(), kernelInfo.args.end(), std::back_inserter(inputTypesUnranked),
                   [](mlir::Attribute arg) {
                       return arg.getType();
                   });

    // first creating management kernel definition
    VPUIP::createRuntimeKernelDefinition(module, log);

    // Create built-in function ------------------------------------------------

    SmallString builtInFunctionName{"builtin_actshave"};

    auto builtInFunction = VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypesUnranked,
                                                        kernelInfo.entryName, kernelInfo.sourceFileName, log);

    //  Function ---------------------------------------------------------------

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(),
                                             llvm::formatv("actshave_{0}_{1}", inputType, outputType).str(), funcType,
                                             builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    //  Build main function: input/Output cmx ----------------------------------

    const auto inputcmx_offset = 0;
    const auto outputcmx_offset = inputcmx_offset + totalTensorSize(inShape, inputType);

    auto inputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, inShape, inputType, DimsOrder::NHWC);
    std::cout << llvm::formatv("funcinput {0} \n inputcmx_type: {1}", funcinput.getType(), inputcmx_type).str()
              << std::endl;
    vpux::VPURT::DeclareBufferOp inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, inputcmx_offset);

    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, outShape, outputType, DimsOrder::NHWC);
    vpux::VPURT::DeclareBufferOp outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, outputcmx_offset);

    //  Build main function: barriers ------------------------------------------

    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // Spawn Task: Load --------------------------------------------------------

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                builder.getUnknownLoc(), funcinput, getTensorResult(inputcmx));

    // Spawn Task: Kernel ------------------------------------------------------

    auto kernelBuilder = [&](auto /*fn object*/ kernelTaskBody) {
        auto taskOp = funcbuilder.create<vpux::VPURT::TaskOp>(funcbuilder.getUnknownLoc(),
                                                              mlir::ValueRange(barrier0.barrier()),
                                                              mlir::ValueRange(barrier1.barrier()));

        mlir::OpBuilder::InsertPoint lastInsertionPoint = funcbuilder.saveInsertionPoint();
        auto& block = taskOp.body().emplaceBlock();
        funcbuilder.setInsertionPointToStart(&block);

        kernelTaskBody();

        funcbuilder.restoreInsertionPoint(lastInsertionPoint);
    };

    kernelBuilder([&]() {
        // TODO : tile 0
        const int64_t tileIndex = 0;

        auto swKernelOp =
                funcbuilder.create<VPUIP::SwKernelOp>(funcbuilder.getUnknownLoc(), inputcmx.buffer(),
                                                      outputcmx.buffer(), builtInFunction, getIntAttr(ctx, tileIndex));
        VPUIP::initSwKernel(swKernelOp, inputcmx.buffer(), outputcmx.buffer(), kernelInfo.args, log);
    });

    // Spawn Task: Store -------------------------------------------------------

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                builder.getUnknownLoc(), getTensorResult(outputcmx), funcoutput);

    funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    //  Pass Manager -----------------------------------------------------------

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::ReferenceHW, 1, log));
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    //  CNN Operation ----------------------------------------------------------

    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
