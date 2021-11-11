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

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux;

namespace {

//
// ConvertLayers2VPUIPPass
//

class ConvertSWLayers2VPUIPPass final : public ConvertSWLayers2VPUIPBase<ConvertSWLayers2VPUIPPass> {
public:
    explicit ConvertSWLayers2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// Any-SWLayerRewrite
//
class SWLayerRewriter {
public:
    SWLayerRewriter(mlir::MLIRContext* ctx, mlir::Operation* origOp, mlir::PatternRewriter& rewriter, Logger log,
                    mlir::ModuleOp mainModule, mlir::ValueRange inputs, mlir::ValueRange outputs,
                    mlir::ValueRange output_bufs, mlir::ArrayRef<mlir::Attribute> args, mlir::StringRef entryPoint,
                    mlir::StringRef kernelSource)
            : _ctx(ctx),
              _origOp(origOp),
              _log(log),
              _mainModule(mainModule),
              _rewriter(rewriter),
              _inputs(inputs),
              _outputs(outputs),
              _output_bufs(output_bufs),
              _args(args),
              _entryPoint(entryPoint),
              _kernelSource(kernelSource) {
    }

public:
    void rewrite() const {
        auto ctx = _rewriter.get().getContext();

        auto builtInFunction = createBuiltInFunction();

        // TODO : tile 0
        const int64_t tileIndex = 0;
        SmallVector<mlir::Value, 128> inputCMXTensors;
        SmallVector<mlir::Value, 128> outputCMXTensors;
        SmallVector<mlir::Value, 128> outputDmaResults;

        //  creating input dma
        for (auto&& source : _inputs) {
            auto cmxTensor = createCMXTensor(source);
            auto copyOp = _rewriter.get().create<IERT::CopyOp>(_origOp->getLoc(), source, cmxTensor.memref());
            inputCMXTensors.push_back(copyOp.output());
        }

        // allocating output tensors
        for (auto&& output_buf : _output_bufs) {
            auto allocOp = createCMXTensor(output_buf);
            outputCMXTensors.push_back(allocOp.memref());
        }

        mlir::Type integerType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);

        auto sw_kernel_op = _rewriter.get().create<VPUIP::SW_KernelOp>(_origOp->getLoc(), inputCMXTensors,
                                                                       outputCMXTensors, builtInFunction,
                                                                       mlir::IntegerAttr::get(integerType, tileIndex));

        initSwKernel(sw_kernel_op, inputCMXTensors, outputCMXTensors);

        //  creating post-dma
        std::transform(_output_bufs.begin(), _output_bufs.end(), sw_kernel_op.results().begin(),
                       std::back_inserter(outputDmaResults), [&](const auto& output_buf, const auto& swKernelResult) {
                           auto copyOp =
                                   _rewriter.get().create<IERT::CopyOp>(_origOp->getLoc(), swKernelResult, output_buf);
                           return copyOp.output();
                       });

        // setting output to be from DMA
        _rewriter.get().replaceOp(_origOp, outputDmaResults);
    }

private:
    mlir::memref::AllocOp createCMXTensor(mlir::Value source) const {
        auto type = source.getType().template dyn_cast<mlir::MemRefType>();

        const auto cmxMemSpaceAttr =
                VPUIP::PhysicalMemoryAttr::get(_rewriter.get().getContext(), VPUIP::PhysicalMemory::CMX_NN);
        const auto dataTypeCMX = changeMemSpace(type, cmxMemSpaceAttr);

        // TODO : how tile index should be used ???
        return _rewriter.get().create<mlir::memref::AllocOp>(source.getLoc(), dataTypeCMX);
    }

    void initSwKernel(VPUIP::SW_KernelOp& sw_kernel_op, mlir::ValueRange inputs, mlir::ValueRange output_bufs) const {
        OpBuilderLogger builderLog(_log.nest());
        auto& bodyRegion = sw_kernel_op.body();
        auto& sw_kernel_block = bodyRegion.emplaceBlock();

        // embedding block args
        auto addBlockArgs = [&sw_kernel_block](auto& cnt) {
            for (auto&& arg : cnt) {
                sw_kernel_block.addArgument(arg.getType());
            }
        };
        addBlockArgs(inputs);
        addBlockArgs(output_bufs);

        auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&sw_kernel_block, &builderLog);

        // embedding args of IERT operation as constants

        llvm::SmallVector<mlir::arith::ConstantOp, 12> constantArgs;
        for (auto&& arg : _args) {
            constantArgs.push_back(
                    swKernelBlockBuilder.template create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(_ctx), arg));
        }

        // pack input/outputs and constants into single call to sw_kernel_run
        llvm::SmallVector<mlir::Value, 12> operands;
        auto fetchOperands = [&operands](auto& cnt) {
            for (auto&& arg : cnt) {
                operands.push_back(arg);
            }
        };
        auto blockArgs = sw_kernel_block.getArguments();
        fetchOperands(blockArgs);
        fetchOperands(constantArgs);

        swKernelBlockBuilder.template create<VPUIP::SW_Kernel_run>(mlir::UnknownLoc::get(_ctx),
                                                                   mlir::ValueRange(operands));
    }

    mlir::SymbolRefAttr createBuiltInFunction() const {
        auto mainModuleLoc = _mainModule;
        vpux::OpBuilderLogger builderLog(_log.nest());

        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(mainModuleLoc.getBody(), &builderLog);
        auto innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(_ctx), StringRef("VPU.SW"));

        auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(innerModule.getBody(), &builderLog);

        llvm::SmallString<128> built_in_name{"builtin_"};
        auto nonNamespaceOpName = _origOp->getName().getStringRef().slice(
                _origOp->getName().getDialectNamespace().size() + 1, mlir::StringRef::npos);
        built_in_name.append(nonNamespaceOpName);

        auto builtInFunctionInternal = mlir::SymbolRefAttr::get(_ctx, built_in_name);

        auto builtInFunction =
                mlir::SymbolRefAttr::get(_ctx, innerModule.getName().getValue(), {builtInFunctionInternal});

        mlir::SmallVector<mlir::Type, 12> inputTypes;

        auto fetchByType = [&inputTypes](auto& cnt) {
            for (auto&& arg : cnt) {
                inputTypes.template emplace_back<mlir::Type>(arg.getType());
            }
        };
        fetchByType(_inputs);
        fetchByType(_outputs);
        fetchByType(_args);

        const auto funcType = mlir::FunctionType::get(_ctx, inputTypes, mlir::TypeRange{});

        auto buildInOp =
                innerModuleBuilder.template create<mlir::FuncOp>(mlir::UnknownLoc::get(_ctx), built_in_name, funcType);

        // modifying attributes
        buildInOp.sym_visibilityAttr(mlir::StringAttr::get(_ctx, "private"));

        buildInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(_ctx, _entryPoint));
        buildInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(_ctx, _kernelSource));

        return builtInFunction;
    }

    mlir::MLIRContext* _ctx;
    mlir::Operation* _origOp;
    Logger _log;
    mlir::ModuleOp _mainModule;
    std::reference_wrapper<mlir::PatternRewriter> _rewriter;
    mlir::ValueRange _inputs;
    mlir::ValueRange _outputs;
    mlir::ValueRange _output_bufs;
    mlir::ArrayRef<mlir::Attribute> _args;
    mlir::StringRef _entryPoint;
    mlir::StringRef _kernelSource;
};

class RewriteSoftmaxMTL final : public mlir::OpRewritePattern<IERT::SoftMaxOp> {
public:
    RewriteSoftmaxMTL(mlir::MLIRContext* ctx, Logger log, mlir::ModuleOp mainModule)
            : mlir::OpRewritePattern<IERT::SoftMaxOp>(ctx), _log(log), _mainModule(mainModule) {
    }
    mlir::LogicalResult matchAndRewrite(IERT::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final {
        mlir::SmallVector<mlir::Attribute, 128> args = {origOp.axisIndAttr()};
        SWLayerRewriter(getContext(), origOp.getOperation(), rewriter, _log, _mainModule, {origOp.input()},
                        {origOp.output()}, {origOp.output_buff()}, args, "softmax_fp16", "softmax_fp16.cpp")
                .rewrite();
        return mlir::success();
    }

protected:
    Logger _log;
    mlir::ModuleOp _mainModule;
};

class RewriteSigmoidMTL final : public mlir::OpRewritePattern<IERT::SigmoidOp> {
public:
    RewriteSigmoidMTL(mlir::MLIRContext* ctx, Logger log, mlir::ModuleOp mainModule)
            : mlir::OpRewritePattern<IERT::SigmoidOp>(ctx), _log(log), _mainModule(mainModule) {
    }
    mlir::LogicalResult matchAndRewrite(IERT::SigmoidOp origOp, mlir::PatternRewriter& rewriter) const final {
        SWLayerRewriter(getContext(), origOp.getOperation(), rewriter, _log, _mainModule, {origOp.input()},
                        {origOp.output()}, {origOp.output_buff()}, {}, "sigmoid_fp16", "sigmoid_fp16.cpp")
                .rewrite();
        return mlir::success();
    }

protected:
    Logger _log;
    mlir::ModuleOp _mainModule;
};

void ConvertSWLayers2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPUIP::getArch(module);
    if (arch != VPUIP::ArchKind::MTL) {
        _log.trace("ConvertSWLayers2VPUIPPass enabled only for MTL device, but not for {0}", arch);
        return;
    }

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<Const::DeclareOp, IERT::StaticAllocOp>();
    target.addLegalOp<IERT::SubViewOp, IERT::ConcatViewOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::PermuteCastOp>();
    target.addLegalOp<IERT::QuantizeCastOp>();
    target.addLegalOp<IERT::TimestampOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<IERT::CopyOp>();
    target.addLegalOp<VPUIP::SW_KernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SW_KernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<RewriteSoftmaxMTL>(&ctx, _log, module);
    patterns.insert<RewriteSigmoidMTL>(&ctx, _log, module);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPPass>(log);
}
