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
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"  // 2021_11_08
#include "vpux/compiler/dialect/const/ops.hpp"

#include "llvm/Support/Debug.h"  // Alex: 2021_11_08

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
//#include "mlir/IR/PatternMatch.h"

// clang-format off

using namespace vpux;

namespace {



//
// Generated
//

//#include <vpux/compiler/conversion/rewriters/generated/convert_declarations_to_VPUIP.hpp.inc>

//
// ConvertVPUIP2VPUIPRegMappedPass
//

class ConvertVPUIP2VPUIPRegMappedPass final : public ConvertVPUIP2VPUIPRegMappedBase<ConvertVPUIP2VPUIPRegMappedPass> {
public:
    explicit ConvertVPUIP2VPUIPRegMappedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

class ConvertVPURTDeclareBuffer final : public mlir::OpRewritePattern<VPURT::DeclareBufferOp> {
public:
    ConvertVPURTDeclareBuffer(mlir::MLIRContext* ctx)  //, int64_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPURT::DeclareBufferOp>(ctx)  //, _numDPU(numDPU), _arch(arch), _log(log) {
    {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::DeclareBufferOp origOp, mlir::PatternRewriter& rewriter) const {
        llvm::dbgs() << "Entered ConvertVPURTDeclareBuffer::matchAndRewrite()\n";
        llvm::dbgs().flush();

        rewriter.replaceOpWithNewOp<VPUIPRegMapped::DeclareBufferOp>(
                // rewriter.replaceOpWithNewOp<VPUIP::DeclareBufferOp>(
                origOp, origOp.getOperation()->getResult(0).getType(),
                (vpux::VPUIPRegMapped::MemoryLocation)origOp.locale(),
                // origOp.locale(),
                origOp.dataIndex());  //, nceOp.output(), origOp.output_buff());

        llvm::dbgs() << "Exiting ConvertVPURTDeclareBuffer::matchAndRewrite()\n";
        llvm::dbgs().flush();

        return mlir::success();
    }
};


class ConvertVPURTConfigureBarrier final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrier(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const {
        llvm::dbgs() << "Entered ConvertVPURTConfigureBarrier::matchAndRewrite()\n";
        llvm::dbgs().flush();

        //(void)origOp;
        //(void)rewriter;

        mlir::ValueRange waitBarriers, updateBarriers;  // MEGA-TODO: put right values

        // Inspired from build-x86_64/src/vpux_compiler/include/vpux/compiler/dialect/VPUIPRegMapped/generated/types.cpp.inc, static ::mlir::OptionalParseResult generatedTypeParser()
        // vpux::VPUIPRegMapped::BarrierType bType = vpux::VPUIPRegMapped::BarrierType::get(getContext()); // 2021_11_30
        vpux::VPURT::BarrierType bType = vpux::VPURT::BarrierType::get(getContext()); // 2021_11_30

        // Using this builder:
        // void ConfigureBarrierOp::build(::mlir::OpBuilder &odsBuilder,
        // ::mlir::OperationState &odsState, ::mlir::Type barrier,
        // int64_t id,
        // uint16_t next_same_id,
        // ::mlir::ValueRange waitBarriers, ::mlir::ValueRange updateBarriers)
        rewriter.replaceOpWithNewOp<VPUIPRegMapped::ConfigureBarrierOp>(
                origOp,
                bType, //origOp.getOperation()->getResult(0).getType(),
                origOp.id(),
                // MEGA-TODO: put also a virtualId attribute in ConfigureBarrierOp, as it is found in VPURT::ConfigureBarrierOp
                1, // MEGA-TODO: put right value // uint16_t next_same_id
                // See https://mlir.llvm.org/doxygen/classmlir_1_1ValueRange.html
                // Note: from https://mlir.llvm.org/doxygen/Operation_8h_source.html: using operand_range = OperandRange;
                waitBarriers, // mlir::ValueRange(origOp.waitBarriers()), // MEGA-TODO
                updateBarriers // mlir::ValueRange(origOp.updateBarriers()) // MEGA-TODO
                //waitBarriers, updateBarriers
                );

        return mlir::success();
    }
};


class ConvertVPURTTask final : public mlir::OpRewritePattern<VPURT::TaskOp> {
public:
    ConvertVPURTTask(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPURT::TaskOp>(ctx) {
        llvm::dbgs() << "Entered the constructor of ConvertVPURTTask\n";
        llvm::dbgs().flush();
    }

    mlir::LogicalResult matchAndRewrite(VPURT::TaskOp origOp, mlir::PatternRewriter& rewriter) const {
        llvm::dbgs() << "Entered ConvertVPURTTask::matchAndRewrite()\n";
        llvm::dbgs().flush();

        /*
        // From https://github.com/llvm/llvm-project/blob/48cd5b72b13c1283eedb0f3fac7c14167da7fc2f/mlir/lib/Dialect/StandardOps/Transforms/FuncConversions.cpp#L18
        FunctionType type = callOp.getCalleeType();

        // Convert the original function results.
        SmallVector<Type, 1> convertedResults;
        if (failed(converter.convertTypes(type.getResults(), convertedResults)))
          return failure();
        */

        //(void)origOp;
        //(void)rewriter;

        // mlir::Operation::operand_range waitBarriers();
        // mlir::Operation::operand_range updateBarriers();
        // mlir::UnitAttr compression = origOp.compressionAttr();

        // mlir::TypeRange tr; // small-MEGA TODO: although not important (since it works) maybe try to initialize with a value
        // mlir::ValueRange vr;

        /*
        mlir::IntegerAttr port = origOp.portAttr();
        mlir::IntegerAttr start_after = origOp.start_afterAttr();

        //-->void NNDMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input,
        //::mlir::Value output_buff, ::mlir::ValueRange waitBarriers, ::mlir::ValueRange updateBarriers,
        //bool compression, int64_t port, uint64_t start_after)
        rewriter.replaceOpWithNewOp<VPUIPRegMapped::NNDMAOp>(origOp, origOp.input(), origOp.output_buff(), waitBarriers,
                                                             updateBarriers, compression, port, start_after);
        */

        /*
        // Using this builder:
        // void Task2Op::build(::mlir::OpBuilder &odsBuilder,
        //  ::mlir::OperationState &odsState,
        //  ::mlir::TypeRange resultTypes,
        //  ::mlir::ValueRange waitBarriers,
        //  ::mlir::ValueRange updateBarriers,
        //  bool isTrailingSWLayer)
        VPUIPRegMapped::TaskWithSubregOp aTRMOp = rewriter.replaceOpWithNewOp<VPUIPRegMapped::TaskWithSubregOp>(
                origOp,
                tr, // origOp.getOperation()->getResult(0).getType(),
                mlir::ValueRange(origOp.waitBarriers()),
                mlir::ValueRange(origOp.updateBarriers()),
                origOp.isTrailingSWLayer());
        */

        mlir::Region& aRegOrig = origOp.op();
        mlir::Block& blkOrig = aRegOrig.front();


        /*
        From TimiCompiler Dec 2 2021:
          - the first NNDMAOp takes the waits field from the VPURT.Task op
          - the last NNDMAOp takes the updates field from the VPURT.Task op
          - all the others are with waits and updates on 0.
        */
        int countNNDMAOps = 0;
        for (mlir::Operation& opB : blkOrig) {
            VPUIP::NNDMAOp nndmaOpOrig = llvm::dyn_cast<VPUIP::NNDMAOp>(opB);
            if (nndmaOpOrig) {
                countNNDMAOps++;
            }
        }
        llvm::dbgs() << "ConvertVPURTTask::matchAndRewrite(): countNNDMAOps = " << countNNDMAOps << "\n";
        llvm::dbgs().flush();
        //
        int indexNNDMAOp = 0;
        // Operation &nndmaOpOrig = blkOrig.front();
        for (mlir::Operation& opB : blkOrig) {
            VPUIP::NNDMAOp nndmaOpOrig = llvm::dyn_cast<VPUIP::NNDMAOp>(opB);
            if (nndmaOpOrig) {
                /*
                // From https://mlir.llvm.org/doxygen/classmlir_1_1Region.html
                mlir::BlockAndValueMapping bvm;
                origOp.op().cloneInto(&aReg, bvm);
                */

                /*
                mlir::Region& aReg = aTRMOp.op();

                // mlir::Block& blk2 = origOp.op().front();
                // mlir::Block* blk = &(aReg.front());
                mlir::Block* blkNew = new mlir::Block();

                // Alex: This instruction has to be before defining builderBlk to avoid SegFault
                aReg.push_back(blkNew);

                // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
                mlir::OpBuilder builderBlk(blkNew, blkNew->begin());

                // Using this builder:
                // void NNDMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                // ::mlir::Type output, ::mlir::Value input, ::mlir::Value output_buff,
                // ::mlir::ValueRange waitBarriers, ::mlir::ValueRange updateBarriers,
                // bool compression, int64_t port, uint64_t start_after)
                // VPUIPRegMapped::NNDMAOp newOp =
                builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                                     builderBlk.getUnknownLoc(), // op->getLoc(),
                                                     opB.getResult(0).getType(), // nndmaOpOrig.output(),
                                                     nndmaOpOrig.input(),
                                                     nndmaOpOrig.output_buff(),
                                                     mlir::ValueRange(origOp.waitBarriers()),
                                                     mlir::ValueRange(origOp.updateBarriers()),
                                                     false, // compression
                                                     nndmaOpOrig.port(),
                                                     0 // start_after
                                                     );
                */

                rewriter.replaceOpWithNewOp<VPUIPRegMapped::NNDMAOp>(
                         origOp,
                         opB.getResult(0).getType(), // nndmaOpOrig.output(),
                         nndmaOpOrig.input(),
                         nndmaOpOrig.output_buff(),
                         (indexNNDMAOp == 0) ? mlir::ValueRange(origOp.waitBarriers()) : mlir::ValueRange(),
                         (indexNNDMAOp == countNNDMAOps - 1) ? mlir::ValueRange(origOp.updateBarriers()) : mlir::ValueRange(),
                         false, // compression // MEGA-TODO: I guess, unless compression is a RegMapped value (see huf_en in HglCmxDmaConfigBits in src/dialect/VPUIPRegMapped/ops/dma.cpp), I should take out compression from VPUIPRegMapped::NNDMAOp
                         nndmaOpOrig.port(),
                         0 // start_after // MEGA-TODO: initialize
                        );

                indexNNDMAOp++;
                // break;
            }
        }

        return mlir::success();
    }
};


void ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc() {
    // auto& ctx = getContext();
    mlir::MLIRContext *ctx = & (getContext());
    mlir::FuncOp funcOp = getFunction();

    // See https://mlir.llvm.org/doxygen/classmlir_1_1FunctionPass.html
    llvm::dbgs()
            // printIndent()
            << "Function name: " << funcOp.getName() << "\n";
    llvm::dbgs() << "Entered ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc(). funcOp = " << funcOp << "\n";

    // printRegion(*(funcOp.getCallableRegion()));

    // Default is operations are illegal.

    // mlir::ConversionTarget target(ctx);
    mlir::ConversionTarget target(*ctx);
    // target.addLegalDialect<mlir::async::AsyncDialect>();
    // target.addLegalDialect<Const::ConstDialect>();
    // target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    // target.addLegalOp<VPUIPRegMapped::DeclareBufferOp>();
    // target.addLegalOp<VPURT::TaskOp>();
    target.addLegalOp<VPUIP::NNDMAOp>();
    // target.addLegalOp<VPUIP::DeclareBufferOp>();
    //
    // target.addLegalOp<VPUIPRegMapped::ConfigureBarrierOp>();
    // target.addLegalOp<VPUIP::ConfigureBarrierOp>();
    //
    // target.addLegalOp<builtin.func, mlir::ReturnOp>();
    /*
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<IERT::SubViewOp, IERT::ConcatViewOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::PermuteCastOp>();
    target.addLegalOp<IERT::QuantizeCastOp>();
    */

    //mlir::RewritePatternSet patterns(&ctx);
    mlir::RewritePatternSet patterns(ctx);
    // populateWithGenerated(patterns);

    // patterns.insert<ConvertVPUIPDeclareTensor>(&ctx);

    /*
    // Inspired from https://llvm.discourse.group/t/linalg-intermediate-ir-error/4251/4
    auto registry = ctx.getDialectRegistry();
    // registerAllDialects(ctx.getDialectRegistry());
    ctx.allowUnregisteredDialects(true);
    ctx.printOpOnDiagnostic(true);
    // registry.loadAll(&ctx);
    */

    // From https://mlir.llvm.org/doxygen/classmlir_1_1MLIRContext.html
    // To avoid error: <<LLVM ERROR: Building op `VPUIPRegMapped.DeclareTensor` but it isn't registered in this
    // MLIRContext: the dialect may not be loaded or this operation isn't registered by the dialect.>>
    // ctx->loadAllAvailableDialects();
    // ctx->getOrLoadDialect("VPUIPRegMapped");
    // ctx->getOrLoadDialect("ELF");
    // ctx->getOrLoadDialect("VPUIP");

    patterns.insert<ConvertVPURTDeclareBuffer>(ctx);  //, dpuExec.count(), arch, _log);
    //
    patterns.insert<ConvertVPURTConfigureBarrier>(ctx);
    //
    // 2021_11_24: MEGA TODO:
    //patterns.insert<ConvertVPUIPNNDMA>(ctx);
    patterns.insert<ConvertVPURTTask>(ctx);

    if (mlir::failed(mlir::applyFullConversion(funcOp, target, std::move(patterns)))) {
        // if (mlir::failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns), getDefaultGreedyRewriteConfig())))
        signalPassFailure();
    }

    llvm::dbgs() << "safeRunOnFunc(): funcOp = " << funcOp << "\n";
    llvm::dbgs().flush();

    // printRegion(*(funcOp.getCallableRegion()));
}

}  // namespace

//
// createConvertVPUIP2VPUIPRegMappedPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUIPRegMappedPass(Logger log) {
    return std::make_unique<ConvertVPUIP2VPUIPRegMappedPass>(log);
}

// clang-format on
