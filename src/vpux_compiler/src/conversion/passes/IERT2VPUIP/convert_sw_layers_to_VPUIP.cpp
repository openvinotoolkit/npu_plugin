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
#include "vpux/utils/core/logger.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_sw_layers_to_VPUIP.hpp.inc>

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

// base trait - works for simplest SW kernels
template <class T>
class SW_Kernel_Args_TraitBase {
protected:
    T _actual;
public:
    SW_Kernel_Args_TraitBase(T actual) : _actual(actual) {}
    mlir::SmallVector<mlir::Value, 128> inputs()  {
        return {_actual.input()};
    }
    mlir::SmallVector<mlir::Value, 128> outputs()  {
        return {_actual.output()};
    }
    mlir::SmallVector<mlir::Value, 128> output_buf()  {
        return {_actual.output_buff()};
    }
};


template <class T>
class SW_Kernel_Args_Trait : public SW_Kernel_Args_TraitBase<T> {

public:
    SW_Kernel_Args_Trait(T actual) : SW_Kernel_Args_TraitBase<T>(actual) {}
};

// specialisation for extracting args per given type
template<>
class SW_Kernel_Args_Trait<IERT::SoftMaxOp> : public SW_Kernel_Args_TraitBase<IERT::SoftMaxOp> {
public:
    SW_Kernel_Args_Trait(IERT::SoftMaxOp actual) : SW_Kernel_Args_TraitBase<IERT::SoftMaxOp>(actual) {}
    mlir::SmallVector<mlir::Attribute, 128> args() {
        return {_actual.axisIndAttr()};
    };
    mlir::SmallString<64> entryPoint() const {
        return {"softmax_fp16"};
    }
    mlir::SmallString<64> source() const {
        return {"softmax_fp16.cpp"};
    }
};

template<>
class SW_Kernel_Args_Trait<IERT::SigmoidOp> : public SW_Kernel_Args_TraitBase<IERT::SigmoidOp> {
public:
    SW_Kernel_Args_Trait(IERT::SigmoidOp actual) : SW_Kernel_Args_TraitBase<IERT::SigmoidOp>(actual) {}
    mlir::SmallVector<mlir::Attribute, 128> args() {
        return {};
    };
    mlir::SmallString<64> entryPoint() const {
        return {"sigmoid_fp16"};
    }
    mlir::SmallString<64> source() const {
        return {"sigmoid_fp16.c"};
    }
};


#define TRAIT_NAME(invocation) SW_Kernel_##invocation

#define GEN_TRAIT(result_type, invocation)\
class TRAIT_NAME(invocation) {\
public:\
    template <class T>\
    static result_type Invoke(T op) {\
        SW_Kernel_Args_Trait<T> tmp(op);\
        return tmp.invocation();\
    }\
}

using SmallValueVector = mlir::SmallVector<mlir::Value, 128>;
using SmallAttributeVector = mlir::SmallVector<mlir::Attribute, 128>;

GEN_TRAIT(SmallValueVector, inputs);
GEN_TRAIT(SmallValueVector, outputs);
GEN_TRAIT(SmallValueVector, output_buf);
GEN_TRAIT(SmallAttributeVector, args);
GEN_TRAIT(mlir::SmallString<64>, entryPoint);
GEN_TRAIT(mlir::SmallString<64>, source);


template <class TN, class ... TN_1>
class run_for_type {
public:
    template <class Functor>
    static auto findAndRun(mlir::Operation* op) -> decltype(Functor::template Invoke<TN>(nullptr)) {
        if (auto casted = mlir::dyn_cast_or_null<TN>(op)) {
            return Functor::template Invoke<TN>(casted);
        }
        return run_for_type<TN_1...>::template findAndRun<Functor>(op);
    }
};

template <class T0>
class run_for_type<T0> {
public:
    template <class Functor>
    static auto findAndRun(mlir::Operation* op) -> decltype(Functor::template Invoke<T0>(nullptr)) {
        return Functor::Invoke(mlir::dyn_cast_or_null<T0>(op));
    }
};

template <class ... TN>
class supported_types {
    class FindResult {
    public:
        template <class T>
        static  bool Invoke(T op) {
            return op != nullptr;
        }
    };
public:
    bool isSupported(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<FindResult>(origOp);
    }
    mlir::SmallVector<mlir::Value, 128> inputs(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<TRAIT_NAME(inputs)>(origOp);
    }
    mlir::SmallVector<mlir::Value, 128> outputs(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<TRAIT_NAME(outputs)>(origOp);
    }
    mlir::SmallVector<mlir::Value, 128> output_buf(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<TRAIT_NAME(output_buf)>(origOp);
    }
    mlir::SmallVector<mlir::Attribute, 128> args(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<TRAIT_NAME(args)>(origOp);
    }
    mlir::SmallString<64> entryPoint(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<TRAIT_NAME(entryPoint)>(origOp);
    }
    mlir::SmallString<64> source(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<TRAIT_NAME(source)>(origOp);
    }
};

//
// Any-SWLayerRewrite
//
template <class ... T>
class ANYSWLayerRewrite final : public mlir::RewritePattern {
public:
    ANYSWLayerRewrite(mlir::MLIRContext* ctx, Logger log, mlir::ModuleOp mainModule)
            : mlir::RewritePattern(MatchAnyOpTypeTag{}, mlir::PatternBenefit{1}, ctx), _log(log), mainModule(mainModule) {
    }

    static const supported_types<T ...>& operation() {
        static const supported_types<T ...> registeredOps;
        return registeredOps;
    }

public:
    mlir::LogicalResult match(mlir::Operation *op) const override {
        if (operation().isSupported(op)) {
            op->dump();
            return mlir::success();
        }
        _log.error("unsupported sourceOP : {0}", op->getName());
        return mlir::failure();
    }

    void rewrite(mlir::Operation *origOp, mlir::PatternRewriter &rewriter) const override {
        auto inputs      = operation().inputs(origOp);
        auto outputs     = operation().outputs(origOp);
        auto output_bufs = operation().output_buf(origOp);
        auto args        = operation().args(origOp);

        auto builtInFunction = createBuiltInFunction(origOp, inputs, outputs, args);

        // TODO : tile 0
        const int64_t tileIndex = 0;
        SmallVector<mlir::Value, 128> inputCMXTensors;
        SmallVector<mlir::Value, 128> outputCMXTensors;
        SmallVector<mlir::Value, 128> outputDmaResults;

        createInOutDma(origOp, rewriter, inputs, output_bufs, tileIndex, inputCMXTensors, outputCMXTensors,
                       outputDmaResults);

        mlir::Type integerType =
                mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Unsigned);

        auto sw_kernel_op = rewriter.create<VPUIP::SW_KernelOp>(origOp->getLoc(),
            inputCMXTensors,
            outputCMXTensors,
            builtInFunction,
            mlir::IntegerAttr::get(integerType, tileIndex));

        initSwKernel(inputCMXTensors, outputCMXTensors, args, sw_kernel_op);

        // setting output to be from DMA
        rewriter.replaceOp(origOp, outputDmaResults);
    }
    void createInOutDma(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, SmallVector<mlir::Value, 128>& inputs,
                        SmallVector<mlir::Value, 128>& output_bufs, const int64_t tileIndex,
                        SmallVector<mlir::Value, 128>& inputCMXTensors, SmallVector<mlir::Value, 128>& outputCMXTensors,
                        SmallVector<mlir::Value, 128>& outputDmaResults) const {  // direction true = to CMX
        // direction false = from CMX
        auto createDMA = [&](mlir::Value source, int64_t data_index, bool bDirection) {
            auto type = source.getType().template dyn_cast<mlir::MemRefType>();
            auto cmxType =
                    mlir::MemRefType::get(type.getShape(), type.getElementType(), {},
                                          VPUIP::MemoryLocationAttr::get(getContext(), VPUIP::MemoryLocation::VPU_CMX_NN));

            auto cmxTensorOp = rewriter.create<VPUIP::DeclareTensorOp>(origOp->getLoc(),
                                                           cmxType,
                                                           VPUIP::MemoryLocation::VPU_CMX_NN,
                                                           tileIndex,
                                                           data_index);  // where to get data index ???

            if (bDirection) {
                return rewriter.create<VPUIP::NNDMAOp>(origOp->getLoc(), source, cmxTensorOp.memory());
            }else {
                return rewriter.create<VPUIP::NNDMAOp>(origOp->getLoc(), cmxTensorOp.memory(), source);
            }
        };
        for (auto && inOperand : inputs) {
            auto dma = createDMA(inOperand, 0, true);
            inputCMXTensors.push_back(((VPUIP::NNDMAOp)dma).output_buff());
        }
        for (auto && output_buf : output_bufs) {
            auto dma = createDMA(output_buf, 2000, false);
            outputCMXTensors.push_back(dma.input());
            // dmas all have 1-to-1 copy
            outputDmaResults.push_back(((VPUIP::NNDMAOp)dma).output());
        }
    }

private:

    void initSwKernel(const SmallVector<mlir::Value, 128>& inputs, const SmallVector<mlir::Value, 128>& output_bufs,
                      SmallVector<mlir::Attribute, 128>& args, VPUIP::SW_KernelOp& sw_kernel_op) const {
        OpBuilderLogger builderLog(_log.nest());
        auto ctx = getContext();
        auto& bodyRegion = sw_kernel_op.body();
        auto& sw_kernel_block = bodyRegion.emplaceBlock();

        // embedding block args
        auto addBlockArgs = [&sw_kernel_block](auto &cnt) {
            for (auto &&arg : cnt) {
                sw_kernel_block.addArgument(arg.getType());
            }
        };
        addBlockArgs(inputs);
        addBlockArgs(output_bufs);

        auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&sw_kernel_block, &builderLog);

        // embedding args of IERT operation as constants
        llvm::SmallVector<mlir::ConstantOp, 12> constantArgs;
        for (auto &&arg : args) {
            constantArgs.push_back(swKernelBlockBuilder.template create<mlir::ConstantOp>(mlir::UnknownLoc::get(ctx), arg));
        }

        // pack input/outputs and constants into single call to sw_kernel_run
        llvm::SmallVector<mlir::Value, 12> operands;
        auto fetchOperands = [&operands](auto &cnt) {
            for (auto &&arg : cnt) {
                operands.push_back(arg);
            }
        };
        auto blockArgs = sw_kernel_block.getArguments();
        fetchOperands(blockArgs);
        fetchOperands(constantArgs);

        swKernelBlockBuilder.template create<VPUIP::SW_Kernel_run>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands));
    }


    mlir::SymbolRefAttr createBuiltInFunction(mlir::Operation * origOp, mlir::ValueRange inputs,
                               mlir::ValueRange outputs, llvm::ArrayRef<mlir::Attribute> args) const {

        auto ctx = getContext();
        auto mainModuleLoc = mainModule;
        vpux::OpBuilderLogger builderLog(_log.nest());

        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(mainModuleLoc.getBody(), &builderLog);
        auto innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), StringRef("VPU.SW"));

        auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(innerModule.getBody(), &builderLog);

        llvm::SmallString<128> built_in_name {"builtin_"};
        auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                         mlir::StringRef::npos);
        built_in_name.append(nonNamespaceOpName);

        auto builtInFunctionInternal = mlir::SymbolRefAttr::get(ctx, built_in_name);

        auto builtInFunction = mlir::SymbolRefAttr::get(ctx,
                                                        innerModule.getName().getValue(),
                                                        {builtInFunctionInternal});

        mlir::SmallVector<mlir::Type, 12> inputTypes;

        auto fetchByType = [&inputTypes](auto &cnt) {
            for (auto &&arg : cnt) {
                inputTypes.template emplace_back<mlir::Type>(arg.getType());
            }
        };
        fetchByType(inputs);
        fetchByType(outputs);
        fetchByType(args);

        const auto funcType = mlir::FunctionType::get(ctx, inputTypes, mlir::TypeRange{});

        auto buildInOp = innerModuleBuilder.template create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx),
                                                                          built_in_name,
                                                                          funcType);

        // modifying attributes
        buildInOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));
        auto entryPoint = operation().entryPoint(origOp);
        auto sourceFile  = operation().source(origOp);

        buildInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, entryPoint));
        buildInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, sourceFile));

        return builtInFunction;
    }

    Logger _log;
    mlir::ModuleOp mainModule;
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
    target.addLegalOp<IERT::GenericReshapeOp, IERT::ImplicitReorderOp>();
    target.addLegalOp<IERT::TimestampOp>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<ANYSWLayerRewrite<
            IERT::SigmoidOp,
            IERT::SoftMaxOp>>(&ctx, _log, module);

    // TODO: maybe use this container instead of Variadics
    // if many types in iert has different input/outputs and args description this make sense
    /*patterns.insert<LSTMCellRewrite>(&ctx, _log);
    patterns.insert<LSTMSequenceRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<FullyConnectedRewrite>(&ctx, _log);
    patterns.insert<RewriteConvolution>(&ctx, _log)*/
    populateWithGenerated(patterns);

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
