//
// Copyright (C) 2022 Intel Corporation.
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
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <iostream>

using namespace vpux;

namespace {

//
// ConvertVPUIP2VPUIPRegMappedPass
//

class ConvertVPUIP2VPUIPRegMappedPass final : public ConvertVPUIP2VPUIPRegMappedBase<ConvertVPUIP2VPUIPRegMappedPass> {
public:
    explicit ConvertVPUIP2VPUIPRegMappedPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    Logger _log;

    void replaceVPURTTaskOpWithNNDMAOp(mlir::MLIRContext* ctx, mlir::FuncOp& funcOp, Logger _log) {
        _log.info("Entered replaceVPURTTaskOpWithNNDMAOp().");

        for (;;) {
            bool foundTaskOp = false;

            for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
                foundTaskOp = true;

                _log.info("replaceVPURTTaskOpWithNNDMAOp(): taskOp = {0}", taskOp);

                // mlir::Region& aRegOrig = taskOp.body();
                // mlir::Block& blkOrig = aRegOrig.front();

                // Although on the TimiCompiler meeting from Dec 2nd 2021 we discussed that:
                //  - the first NNDMAOp takes the waits field from the VPURT.TaskOp
                //  - the last NNDMAOp takes the updates field from the VPURT.TaskOp
                //  - all the others are with waits and updates on 0.
                // Currently the VPURT TaskOp can have only 1 block with only 1 Op,
                //    and this op is for now an NNDMAOp.

                for (auto op : taskOp.body().getOps<VPUIP::NNDMAOp>()) {
                    _log.info("replaceVPURTTaskOpWithNNDMAOp(): op = {0}.", op);
                    _log.info("replaceVPURTTaskOpWithNNDMAOp(): op.getNumResults() = {0}.",
                              op.getOperation()->getNumResults());
                    _log.info("replaceVPURTTaskOpWithNNDMAOp(): op.getResult(0).getType() = {0}.",
                              op.getOperation()->getResult(0).getType());

                    mlir::OpBuilder builderBlk(taskOp);

                    auto indexType = VPUIPRegMapped::IndexType::get(ctx, 1);

                    auto wait_bars = taskOp.waitBarriers();
                    auto update_bars = taskOp.updateBarriers();

                    for (auto val : wait_bars){
                        val.setType(indexType);
                    }

                    for (auto val : update_bars){
                        val.setType(indexType);
                    }

                    std::cout<<"OMGGGGG ----- "<<indexType.getValue()<<"\n\n";

                    // Using this builder:
                    // void NNDMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                    //   ::mlir::Type output, ::mlir::Value input, ::mlir::Value output_buff,
                    //   ::mlir::ValueRange waitBarriers, ::mlir::ValueRange updateBarriers,
                    //   bool compression, int64_t port, uint64_t start_after)
                    builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                            builderBlk.getUnknownLoc(),                 // op->getLoc(),
                            indexType,  // op.output(),
                            op.input(), op.output_buff(), 
                            mlir::ValueRange(wait_bars),
                            mlir::ValueRange(update_bars),
                            false,  // compression // TODO: I guess, unless compression is a RegMapped
                                    // value (see huf_en in HglCmxDmaConfigBits in
                                    // src/dialect/VPUIPRegMapped/ops/dma.cpp), I should take out compression
                                    // from VPUIPRegMapped::NNDMAOp
                            op.port(),
                            0  // start_after // TODO: initialize
                    );

                    _log.info("replaceVPURTTaskOpWithNNDMAOp(): funcOp = {0}", funcOp);
                }

                taskOp->erase();
                _log.info("replaceVPURTTaskOpWithNNDMAOp(): After erase(): funcOp = {0}", funcOp);

                break;  // Block iterator gets invalidated after erase().
            }

            if (foundTaskOp == false)
                break;
        }  // End forever loop
    }
};

class ConvertVPURTConfigureBarrierOp final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrierOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const {
        _log.info("Entered ConvertVPURTConfigureBarrierOp::matchAndRewrite().");

        mlir::ValueRange waitBarriers, updateBarriers;  // TODO: put right values

        auto ctx = ConvertVPURTConfigureBarrierOp::getContext();

        auto indexType = VPUIPRegMapped::IndexType::get(ctx, 1);

        mlir::IntegerAttr producer_count;  // TODO: init with special value
        mlir::IntegerAttr consumer_count;  // TODO: init with special value

        rewriter.replaceOpWithNewOp<VPUIPRegMapped::ConfigureBarrierOp>(
                origOp,
                indexType,  // origOp.getOperation()->getResult(0).getType(),
                origOp.id(),
                // TODO: put also a virtualId attribute in ConfigureBarrierOp, as it is found in
                //              VPURT::ConfigureBarrierOp
                1,  // TODO: put right value // uint16_t next_same_id
                // See https://mlir.llvm.org/doxygen/classmlir_1_1ValueRange.html
                // Note: from https://mlir.llvm.org/doxygen/Operation_8h_source.html: using operand_range =
                // OperandRange;
                producer_count,  // origOp.producer_countAttr(),
                consumer_count,  // origOp.consumer_countAttr(),
                waitBarriers,    // mlir::ValueRange(origOp.waitBarriers()), // TODO
                updateBarriers   // mlir::ValueRange(origOp.updateBarriers()) // TODO
        );

        return mlir::success();
    }

private:
    Logger _log;
};

void ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp = getFunction();

    _log.info("Entered ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc(). Function name: {0}.", funcOp.getName());
    _log.info("funcOp = {0}", funcOp);

    // Need to call replaceVPURTTaskOpWithNNDMAOp() before applyFullConversion(), else we normally get
    //      <<error: failed to legalize operation 'VPURT.Task'>>
    // We call this function and not add a conversion pattern because
    //   VPURT.Task has 0 results, while VPUIPRegMapped::NNDMAOp has 1 result,
    //   and the conversion would normally give this error: <<mlir/lib/IR/PatternMatch.cpp:304: void
    //   mlir::RewriterBase::replaceOpWithResultsOfAnotherOp(mlir::Operation*, mlir::Operation*): Assertion
    //   `op->getNumResults() == newOp->getNumResults() && "replacement op doesn't match results of original op"'
    //   failed.>>

    _log.info("ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc(): After replaceVPURTTaskOpWithNNDMAOp(): funcOp = {0}",
              funcOp);

    // Default is operations are illegal.


    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<VPURT::DeclareBufferOp, VPUIP::NNDMAOp, VPURT::TaskOp>();

    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ConvertVPURTConfigureBarrierOp>(ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
    }

    replaceVPURTTaskOpWithNNDMAOp(ctx, funcOp, _log);

    _log.info("End ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc(): funcOp = {0}", funcOp);
}

}  // namespace

//
// createConvertVPUIP2VPUIPRegMappedPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUIPRegMappedPass(Logger log) {
    return std::make_unique<ConvertVPUIP2VPUIPRegMappedPass>(log);
}
