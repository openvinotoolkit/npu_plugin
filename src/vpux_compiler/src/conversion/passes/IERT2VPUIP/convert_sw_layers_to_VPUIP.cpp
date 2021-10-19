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
#include "vpux/compiler//utils/logging.hpp"

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
class SW_Kernel_Args_Trait {
    T _actual;
public:
    SW_Kernel_Args_Trait(T actual) : _actual(actual) {}
    mlir::ValueRange inputs()  {
        return {_actual.input()};
    }
    mlir::ValueRange outputs()  {
        return {_actual.output()};
    }
    mlir::ValueRange args() {
        return {};
          //  return mlir::ValueRange();
    };
};

class SW_Kernel_Inputs {
public:
    template <class T>
    static mlir::ValueRange Invoke(T op) {
        SW_Kernel_Args_Trait<T> tmp(op);
        return tmp.inputs();
    }
};

class SW_Kernel_Outputs {
public:
    template <class T>
    static mlir::ValueRange Invoke(T op) {
        SW_Kernel_Args_Trait<T> tmp(op);
        return tmp.outputs();
    }
};

class SW_Kernel_Args {
public:
    template <class T>
    static mlir::ValueRange Invoke(T op) {
        SW_Kernel_Args_Trait<T> tmp(op);
        return tmp.args();
    }
};



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
    ::mlir::ValueRange inputs(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<SW_Kernel_Inputs>(origOp);
    }
    ::mlir::ValueRange outputs(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<SW_Kernel_Outputs>(origOp);
    }
    ::mlir::ValueRange args(mlir::Operation* origOp) const {
        return run_for_type<TN ...>::template findAndRun<SW_Kernel_Args>(origOp);
    }
};

//
// Any-SWLayerRewrite
//
template <class ... T>
class ANYSWLayerRewrite final : public mlir::RewritePattern {
public:
    ANYSWLayerRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::RewritePattern(MatchAnyOpTypeTag{}, mlir::PatternBenefit{1}, ctx), _log(log) {
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
        auto inputs  = operation().inputs(origOp);
        auto outputs = operation().outputs(origOp);
        auto args    = operation().args(origOp);

        auto builtInFunction = createBuiltInFunction(origOp, inputs, outputs, args);

        rewriter.replaceOpWithNewOp<VPUIP::SW_Kernel>(origOp,
            inputs,
            outputs,
            builtInFunction,  // TODO: add generation of built-in functions into trait
            mlir::IntegerAttr(0), // tile 0
            args);
    }

private:
    mlir::SymbolRefAttr createBuiltInFunction(mlir::Operation * origOp, mlir::ValueRange inputs,
                               mlir::ValueRange outputs, mlir::ValueRange args) const {
        auto ctx = getContext();
        vpux::OpBuilderLogger builderLog(_log.nest());
        auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef("VPU.SW"));
        auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        llvm::SmallString<128> built_in_name {"builtin_"};
        auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                         mlir::StringRef::npos);
        built_in_name.append(nonNamespaceOpName);

        auto builtInFunction = mlir::SymbolRefAttr::get(ctx, built_in_name);

//        SmallVector <mlir::Value, 128> func_args;
//
//        func_args.insert(func_args.end(), inputs.begin(), inputs.end());
//        func_args.insert(func_args.end(), outputs.begin(), outputs.end());
//        func_args.insert(func_args.end(), args.begin(), args.end());
//
//        mlir::ValueRange v1(func_args);
//        mlir::TypeRange inputTypes(v1);
//
//        const auto funcType = mlir::FunctionType::get(ctx, t1, mlir::TypeRange{});

        /*const auto inType = inputs.front().getType().cast<mlir::ShapedType>();
        const auto outType = outputs.front().getType().cast<mlir::ShapedType>();*/


        args = args;

        SmallVector<mlir::Type, 128> inputTypes = {
                inputs.front().getType(),
                outputs.front().getType(),
                mlir::IntegerType::get(ctx, 0, mlir::IntegerType::Signed)
        };

        const auto funcType = mlir::FunctionType::get(ctx, inputTypes, mlir::TypeRange{});

//        SmallVector<mlir::NamedAttribute, 128> attr;
//
//        attr.push_back({"VPU.kernel_code",  "sigmoid_fp16.c"});
//        attr.push_back({"VPU.kernel_entry", "sigmoid_fp16"});

        auto ff = moduleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunction.getValue(), funcType);

        ff = ff;

        return builtInFunction;
    }

    Logger _log;
};


void ConvertSWLayers2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

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
            IERT::SoftMaxOp>>(&ctx, _log);

    // TODO: maybe use this container instead of Variadics
    // if many types in iert has different input/outputs and args description this make sense
    /*patterns.insert<LSTMCellRewrite>(&ctx, _log);
    patterns.insert<LSTMSequenceRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<FullyConnectedRewrite>(&ctx, _log);
    patterns.insert<RewriteConvolution>(&ctx, _log)*/
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
//    mlir::Operation * op;
//    //IERT::SigmoidOp *sop;
//    if (mlir::cast<IERT::SigmoidOp>(*op)) {
//        _log.error("can cast to sigmoid");
//    }
}


}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPPass>(log);
}
