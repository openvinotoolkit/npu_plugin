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

#include "vpux/hwtest/hwtest.hpp"

#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Support/FileUtilities.h>

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux_config.hpp"

// For building IE dialect
#include <numeric>
#include "vpux/utils/IE/float16.hpp"

// For building pipeline
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/pipelines.hpp"

namespace vpux {

mlir::OwningModuleRef importHWTEST(llvm::StringRef sourceJson, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);
    ctx->loadDialect<VPUIP::VPUIPDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef("mainModule"));
    auto log = Logger{"vpux-hwtest", LogLevel::Trace};
    auto builder = mlir::OpBuilder(module.getBodyRegion());
    auto builderLog = OpBuilderLogger{log.nest()};

    //
    // Define Metacommand params
    //
    // const Shape shape(desc.getDims().begin(), desc.getDims().end());
    const Shape shape(
    //    {1, 16, 1, 1}
    //    {1, 16, 32, 32}
        {1, 10, 32, 32}       // unaligned case
    //    {1, 3, 224, 224}      // produces 4 NCEClusterTasks
    );
    const auto precision = mlir::Float16Type::get(ctx);
    const auto order = DimsOrder::NHWC;
    const auto inputTensorType = vpux::getTensorType(shape.raw(), precision, order, nullptr);
    const auto inputNameAttr = mlir::StringAttr::get(ctx, "input");
    const auto inputTypeAttr = mlir::TypeAttr::get(inputTensorType);

    //
    // Define CNNNetworkOp
    //
    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(ctx, "main");
    auto cnnOp = builder.create<IE::CNNNetworkOp>(mlir::UnknownLoc::get(ctx), mainFuncName, false);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    //
    // Define DataInfoOp input
    //
    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());
    inputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), inputNameAttr, inputTypeAttr);

    //
    // Define DataInfoOp output
    //
    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), builder.getListener());
    const auto outputTensorType = vpux::getTensorType(shape.raw(), precision, order, nullptr);
    const auto outputNameAttr = mlir::StringAttr::get(ctx, "output");
    const auto outputTypeAttr = mlir::TypeAttr::get(outputTensorType);
    outputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), outputNameAttr, outputTypeAttr);

    //
    // Define main function
    //
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(1);
    inputTypes.push_back(inputTensorType);

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(1);
    outputTypes.push_back(outputTensorType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), makeArrayRef(outputTypes));
    auto func = builder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), mainFuncName.getValue(), funcType);
    builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    auto funcInput = func.getArgument(0);
    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(1);

    //
    // Define weights
    //
    const llvm::SmallVector<std::int64_t> weightsShape{shape.raw()[1], shape.raw()[1], 1, 1};
    const auto weightsType = vpux::getTensorType(weightsShape, precision, DimsOrder::NCHW, nullptr);
    const auto weightsElements = std::accumulate(weightsShape.begin(), weightsShape.end(), static_cast<std::int64_t>(1),
                                                 std::multiplies<std::int64_t>());
    std::vector<float16> weightsValues(weightsElements, 0);
    // Debug - identity weights
#if 0
    auto getFP16IdentityWeights = [&](int OC) {
        std::vector<float16> identityWeights(OC*OC, 0);
        for (auto h=0; h<OC; ++h) {
            for (auto w=0; w<OC; ++w) {
                identityWeights.at(h*OC+w) = ((h==w) ? 1.0 : 0);
            }
        }
        return identityWeights;
    };
    weightsValues = getFP16IdentityWeights(weightsShape[1]);
#endif
    auto weightsData = mlir::DenseElementsAttr::get(weightsType, llvm::makeArrayRef<float16>(weightsValues));
    auto weightsAttribute = vpux::Const::ContentAttr::get(weightsData);
    auto weightsDDR = builder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsType, weightsAttribute);

    //
    // Define op
    //
    const llvm::SmallVector<std::int64_t> strides = {1, 1};
    const llvm::SmallVector<std::int64_t> padsBegin = {0, 0};
    const llvm::SmallVector<std::int64_t> padsEnd = {0, 0};
    const llvm::SmallVector<std::int64_t> dilation = {1, 1};
    const auto attrStride = getIntArrayAttr(ctx, strides);
    const auto attrPadsBegin = getIntArrayAttr(ctx, padsBegin);
    const auto attrPadsEnd = getIntArrayAttr(ctx, padsEnd);
    const auto attrDilation = getIntArrayAttr(ctx, dilation);
    auto convOp = builder.create<IE::ConvolutionOp>(builder.getUnknownLoc(), funcInput,
                                                    weightsDDR.getOperation()->getResult(0), nullptr, attrStride,
                                                    attrPadsBegin, attrPadsEnd, attrDilation, nullptr);
    auto reluOp = builder.create<IE::ReLUOp>(builder.getUnknownLoc(), convOp);
    auto reorderOp = builder.create<IE::ReorderOp>(builder.getUnknownLoc(), outputTensorType, reluOp,
                                                   order.toPermutationAffineMap(builder.getContext()));

    //
    // Define function output
    //
    funcOutputs.push_back(reorderOp->getResult(0));
    builder.create<mlir::ReturnOp>(mlir::UnknownLoc::get(ctx), llvm::makeArrayRef(funcOutputs));

    llvm::outs() << "\n"; module.dump(); llvm::outs() << "\n";
    //
    // Configure PassManager
    //
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.enableVerifier(/*verifyPasses=*/true);

    const auto shouldPrintBeforePass = [&](mlir::Pass*, mlir::Operation*) {
        return false;
    };
    const auto shouldPrintAfterPass = [&](mlir::Pass*, mlir::Operation*) {
        return true;
    };

    pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                        /*_printFullIR*/false, /*printAfterOnlyOnChange*/false,
                        /*printAfterOnlyOnFailure*/false, /**_irDumpStream*/llvm::outs(),
                        /*flags*/mlir::OpPrintingFlags());

    //
    // Add SetCompileParams pass
    //
    auto archKind = vpux::VPUIP::ArchKind::KMB;
    auto compilationMode = vpux::VPUIP::CompilationMode::ReferenceHW;
    auto numOfDPUGroups = 1;
    pm.addPass(createSetCompileParamsPass(archKind, compilationMode, numOfDPUGroups, log.nest()));

    //
    // Add HW-mode pipeline
    //
    vpux::buildHardwareModePipeline(pm);

    //
    // Run pipeline
    //
    if (failed(pm.run(module))){
        llvm::outs() << "pipeline failed.\n\n";
        return module;
    }

    //
    // Serialize VPUIP dialect
    //
    mlir::DefaultTimingManager tm;
    auto timing = tm.getRootScope();
    auto blob = VPUIP::exportToBlob(module, timing, log);
    std::string err;
    // dump the blob in a file
    std::unique_ptr<llvm::ToolOutputFile> outFile = mlir::openOutputFile("vpuip.blob", &err);
    outFile->os().write(reinterpret_cast<const char*>(blob.data()), blob.size());
    outFile->keep();
    log.info("Saving blob to {0}", outFile->getFilename());

    nb::TestCaseJsonDescriptor jsonDesc(sourceJson);

#if 0
    if (jsonDesc.getCaseType() == nb::CaseType::activationKernelSimple) {
        hwtest::buildActKernelTest(jsonDesc, module, builder, log);
        return module;
    }

    nb::InputLayer input = jsonDesc.getInputLayer();
    nb::OutputLayer output = jsonDesc.getOutputLayer();

    mlir::Type input_type = hwtest::parseInputType(builder, input);
    mlir::Type output_type = hwtest::parseOutputType(builder, output);

    // TODO:
    // This will be handled later based on op type in config json
    auto opType = jsonDesc.getCaseStr();

    bool isConv = jsonDesc.getCaseType() == nb::CaseType::ZMajorConvolution;
    bool isEltwiseAdd = jsonDesc.getCaseType() == nb::CaseType::EltwiseAdd;
    bool isMaxPool = jsonDesc.getCaseType() == nb::CaseType::MaxPool;
    bool isEltwiseMult = jsonDesc.getCaseType() == nb::CaseType::EltwiseMult;
    bool isAvgPool = jsonDesc.getCaseType() == nb::CaseType::AvgPool;

    auto weightType = [&]() {
        nb::WeightLayer weight = jsonDesc.getWeightLayer();
        return hwtest::parseWeightsType(builder, weight);
    };

    auto weightInChannels = [&]() {
        nb::WeightLayer weight = jsonDesc.getWeightLayer();
        return weight.shape[1];
    };

    if (isConv) {
        if (weightInChannels() > 8 * 1024) {
            hwtest::buildContinuedConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
        } else {
            hwtest::buildSimpleZMajorConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
        }
    } else if (isEltwiseAdd) {
        hwtest::buildEltwiseAdd(jsonDesc, module, builder, log, input_type, weightType(), output_type);
    } else if (isEltwiseMult) {
        hwtest::buildEltwiseMultWithDwConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
    } else if (isMaxPool) {
        hwtest::buildMaxPool(jsonDesc, module, builder, log, input_type, output_type);
    } else if (isAvgPool) {
        hwtest::buildAvgpoolWithDwConv(jsonDesc, module, builder, log, input_type, output_type);
    } else {
        VPUX_THROW("Unknown type: {0}", opType);
    }

    // llvm::dbgs() << "Current module: " << mlir::debugString(module);

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    mlir::DefaultTimingManager tm;
    auto timing = tm.getRootScope();
    auto blob = VPUIP::exportToBlob(module, timing, log);
    std::string err;
    // dump the blob in a file
    std::unique_ptr<llvm::ToolOutputFile> outFile = mlir::openOutputFile("vpuip.blob", &err);
    outFile->os().write(reinterpret_cast<const char*>(blob.data()), blob.size());
    outFile->keep();
    log.info("Saving blob to {0}", outFile->getFilename());
#endif

    return module;
}

}  // namespace vpux
