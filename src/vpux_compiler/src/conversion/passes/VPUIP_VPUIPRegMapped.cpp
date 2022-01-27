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
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "llvm/Support/Debug.h"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

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
    ConvertVPURTDeclareBuffer(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPURT::DeclareBufferOp>(ctx) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::DeclareBufferOp origOp, mlir::PatternRewriter& rewriter) const {
        // llvm::dbgs() << "Entered ConvertVPURTDeclareBuffer::matchAndRewrite()\n";
        // llvm::dbgs().flush();

        vpux::VPUIPRegMapped::MemoryLocation locale;
        if (origOp.section() == vpux::VPURT::BufferSection::DDR) {
            locale = vpux::VPUIPRegMapped::MemoryLocation::VPU_DDR_Heap;
        } else if (origOp.section() == vpux::VPURT::BufferSection::CMX_NN) {
            locale = vpux::VPUIPRegMapped::MemoryLocation::VPU_CMX_NN;
        } else if (origOp.section() == vpux::VPURT::BufferSection::CMX_UPA) {
            locale = vpux::VPUIPRegMapped::MemoryLocation::VPU_CMX_UPA;
        }
        int64_t dataIndex = 0;  // TODO: initialize - maybe we should put sectionIndex() here

        // llvm::dbgs() << "ConvertVPURTDeclareBuffer::matchAndRewrite(): byteOffset() = " << origOp.byteOffset() <<
        // "\n"; llvm::dbgs() << "ConvertVPURTDeclareBuffer::matchAndRewrite(): sectionIndex() = " <<
        // origOp.sectionIndex()
        //              << "\n";
        // llvm::dbgs() << "ConvertVPURTDeclareBuffer::matchAndRewrite(): section() = "
        //              << stringifyBufferSection(origOp.section()) << "\n";
        // llvm::dbgs().flush();

        rewriter.replaceOpWithNewOp<VPUIPRegMapped::DeclareBufferOp>(
                origOp, origOp.getOperation()->getResult(0).getType(), locale,
                dataIndex  // TODO: maybe we should put sectionIndex() here
        );

        // llvm::dbgs() << "Exiting ConvertVPURTDeclareBuffer::matchAndRewrite()\n";
        // llvm::dbgs().flush();

        return mlir::success();
    }
};

class ConvertVPURTConfigureBarrier final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrier(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const {
        // llvm::dbgs() << "Entered ConvertVPURTConfigureBarrier::matchAndRewrite()\n";
        // llvm::dbgs().flush();

        // VPUX_UNUSED(origOp);
        // VPUX_UNUSED(rewriter);

        mlir::ValueRange waitBarriers, updateBarriers;  // MEGA-TODO: put right values

        vpux::VPURT::BarrierType bType = vpux::VPURT::BarrierType::get(getContext());  // 2021_11_30

        mlir::IntegerAttr producer_count;  // TODO: init with special value
        mlir::IntegerAttr consumer_count;  // TODO: init with special value

        rewriter.replaceOpWithNewOp<VPUIPRegMapped::ConfigureBarrierOp>(
                origOp,
                bType,  // origOp.getOperation()->getResult(0).getType(),
                origOp.id(),
                // MEGA-TODO: put also a virtualId attribute in ConfigureBarrierOp, as it is found in
                //              VPURT::ConfigureBarrierOp
                1,  // MEGA-TODO: put right value // uint16_t next_same_id
                // See https://mlir.llvm.org/doxygen/classmlir_1_1ValueRange.html
                // Note: from https://mlir.llvm.org/doxygen/Operation_8h_source.html: using operand_range =
                // OperandRange;
                producer_count,  // origOp.producer_countAttr(),
                consumer_count,  // origOp.consumer_countAttr(),
                waitBarriers,    // mlir::ValueRange(origOp.waitBarriers()), // MEGA-TODO
                updateBarriers   // mlir::ValueRange(origOp.updateBarriers()) // MEGA-TODO
        );

        return mlir::success();
    }
};

class ConvertVPURTTask final : public mlir::OpRewritePattern<VPURT::TaskOp> {
public:
    ConvertVPURTTask(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPURT::TaskOp>(ctx) {
        // llvm::dbgs() << "Entered the constructor of ConvertVPURTTask\n";
        // llvm::dbgs().flush();
    }

    mlir::LogicalResult matchAndRewrite(VPURT::TaskOp origOp, mlir::PatternRewriter& rewriter) const {
        // llvm::dbgs() << "Entered ConvertVPURTTask::matchAndRewrite()\n";
        // llvm::dbgs().flush();

        mlir::Region& aRegOrig = origOp.body();
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
        // llvm::dbgs() << "ConvertVPURTTask::matchAndRewrite(): countNNDMAOps = " << countNNDMAOps << "\n";
        // llvm::dbgs().flush();
        //
        int indexNNDMAOp = 0;
        // Operation &nndmaOpOrig = blkOrig.front();
        for (mlir::Operation& opB : blkOrig) {
            VPUIP::NNDMAOp nndmaOpOrig = llvm::dyn_cast<VPUIP::NNDMAOp>(opB);
            if (nndmaOpOrig) {
                rewriter.replaceOpWithNewOp<VPUIPRegMapped::NNDMAOp>(
                        origOp,
                        opB.getResult(0).getType(),  // nndmaOpOrig.output(),
                        nndmaOpOrig.input(), nndmaOpOrig.output_buff(),
                        (indexNNDMAOp == 0) ? mlir::ValueRange(origOp.waitBarriers()) : mlir::ValueRange(),
                        (indexNNDMAOp == countNNDMAOps - 1) ? mlir::ValueRange(origOp.updateBarriers())
                                                            : mlir::ValueRange(),
                        false,  // compression // MEGA-TODO: I guess, unless compression is a RegMapped value (see
                                // huf_en in HglCmxDmaConfigBits in src/dialect/VPUIPRegMapped/ops/dma.cpp), I should
                                // take out compression from VPUIPRegMapped::NNDMAOp
                        nndmaOpOrig.port(),
                        0  // start_after // MEGA-TODO: initialize
                );

                indexNNDMAOp++;
            }
        }

        return mlir::success();
    }
};

void ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp = getFunction();

    // llvm::dbgs() << "Function name: " << funcOp.getName() << "\n";
    // llvm::dbgs() << "Entered ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc(). funcOp = " << funcOp << "\n";

    // Default is operations are illegal.

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<VPUIP::NNDMAOp>();

    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ConvertVPURTDeclareBuffer>(ctx);
    patterns.insert<ConvertVPURTConfigureBarrier>(ctx);
    patterns.insert<ConvertVPURTTask>(ctx);

    if (mlir::failed(mlir::applyFullConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
    }

    // llvm::dbgs() << "ConvertVPUIP2VPUIPRegMappedPass::safeRunOnFunc(): funcOp = " << funcOp << "\n";
    // llvm::dbgs().flush();
}

}  // namespace

//
// createConvertVPUIP2VPUIPRegMappedPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUIPRegMappedPass(Logger log) {
    return std::make_unique<ConvertVPUIP2VPUIPRegMappedPass>(log);
}
