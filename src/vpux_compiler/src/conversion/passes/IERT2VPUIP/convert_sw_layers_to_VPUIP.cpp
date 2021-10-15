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
    mlir::ValueRange inputs() const {
        return {};
        //return {_actual->input()};
    }
    mlir::ValueRange outputs() const {
        return {};
        //return {_actual->output()};
    }
    mlir::ValueRange args() const {
        return {};
          //  return mlir::ValueRange();
    };
};

template <>
class SW_Kernel_Args_Trait<mlir::Operation*> {
    mlir::Operation *_actual;
public:
    SW_Kernel_Args_Trait(mlir::Operation * actual) : _actual(actual) {}
    mlir::ValueRange inputs() const {
        return {};
       // static_assert(false, "Operation trait not needed");
    }
    mlir::ValueRange outputs() const {
        return {};
        //static_assert(false, "Operation trait not needed");
    }
    mlir::ValueRange args() const {
        return {};
        //static_assert(false, "Operation trait not needed");
    };
};

class SW_Kernel_Inputs {
public:
    template <class T>
    static mlir::ValueRange Invoke(T op) {
        return SW_Kernel_Args_Trait<T>(op).inputs();
    }
};

class SW_Kernel_Outputs {
public:
    template <class T>
    static mlir::ValueRange Invoke(T op) {
        return SW_Kernel_Args_Trait<T>(op).outputs();
    }
};

class SW_Kernel_Args {
public:
    template <class T>
    static mlir::ValueRange Invoke(T op) {
        return SW_Kernel_Args_Trait<T>(op).args();
    }
};



template <class TN, class ... TN_1>
class run_for_type {
public:
    template <class Functor>
    static auto findAndRun(mlir::Operation* op) -> decltype(Functor::template Invoke<TN*>(nullptr)) {
        if (auto casted = mlir::dyn_cast_or_null<TN>(op)) {
            return Functor::Invoke(casted);
        }
        return run_for_type<TN_1...>::template findAndRun<Functor>(op);
    }
};

template <class T0>
class run_for_type<T0> {
public:
    template <class Functor>
    static auto findAndRun(mlir::Operation* op) -> decltype(Functor::template Invoke<T0*>(nullptr)) {
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


class ANYSWLayerRewrite final : public mlir::RewritePattern {
public:
    ANYSWLayerRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::RewritePattern(MatchAnyOpTypeTag{}, mlir::PatternBenefit{1}, ctx), _log(log) {
    }

    using  sup_types = supported_types<
            IERT::SigmoidOp,
            IERT::SoftMaxOp>;


    static const sup_types& operation() {
        static const sup_types registeredOps;
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
        rewriter.replaceOpWithNewOp<VPUIP::SW_Kernel>(origOp,
            operation().inputs(origOp),
            operation().outputs(origOp),
            mlir::SymbolRefAttr(),  // TODO: add generation of built-in functions into trait
            mlir::IntegerAttr(0), // tile 0
            operation().args(origOp));
    }

private:
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
    patterns.insert<ANYSWLayerRewrite>(&ctx, _log);
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
}


}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPPass>(log);
}
