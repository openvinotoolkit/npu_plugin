//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/Support/FileSystem.h>

#include <kernels/inc/common_types.h>

#include <iostream>
#include <vector>

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
    void safeRunOnModule() final;

    Logger _log;

    template <class T>
    void appendValueToVector(std::vector<uint8_t>& vec, const T& anyValue) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&anyValue), sizeof(anyValue));
        vec.insert(vec.end(), valueAsArray.begin(), valueAsArray.end());
    }

    void addBasicAttrToVector(std::vector<uint8_t>& vec, mlir::Attribute attr) {
        if (auto val = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
            appendValueToVector(vec, val.getValue().getSExtValue());
        } else if (auto val = attr.dyn_cast_or_null<mlir::FloatAttr>()) {
            appendValueToVector(vec, static_cast<float>(val.getValue().convertToDouble()));
        } else {
            VPUX_THROW("Act Shave Invocation: cannot store arg of type {0}", attr.getType());
        }
    }

    void addAttrsToVector(std::vector<uint8_t>& vec, mlir::Attribute attr) {
        if (auto arr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
            auto vals = arr.getValue();
            for (auto val : vals) {
                addBasicAttrToVector(vec, val);
            }
        } else {
            addBasicAttrToVector(vec, attr);
        }
    }

    void addTensorArgToVector(std::vector<uint8_t>& vec, mlir::Value value) {
        sw_params::MemRefData memrefData{};

        const auto shape = getShape(value);
        memrefData.numDims = checked_cast<uint32_t>(shape.size());

        // order
        const auto inOrder = DimsOrder::fromValue(value);
        const auto memShape = inOrder.toMemoryOrder(shape);
        memrefData.dimsOrder = inOrder.invertedCode();

        memrefData.dataType = 0;  // TODO: to be defined
        memrefData.location = sw_params::NN_CMX;

        appendValueToVector(vec, memrefData);
    }

    void replaceVPURTTaskOpWithNNDMAOp(mlir::MLIRContext* ctx, mlir::FuncOp& funcOp, Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithNNDMAOp()");

        auto dma_count = 0;

        mlir::Value previousDMA;

        for (;;) {
            bool foundTaskOp = false;

            for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
                if (!taskOp.body().getOps<VPUIP::NNDMAOp>().empty()) {
                    foundTaskOp = true;
                } else if (!taskOp.body().getOps<VPUIP::SwKernelOp>().empty()) {
                    continue;
                }

                for (auto op : taskOp.body().getOps<VPUIP::NNDMAOp>()) {
                    mlir::OpBuilder builderBlk(taskOp);

                    auto indexType = VPUIPRegMapped::IndexType::get(ctx, dma_count);

                    auto wait_bars = taskOp.waitBarriers();
                    auto update_bars = taskOp.updateBarriers();

                    auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

                    for (auto val : wait_bars) {
                        val.setType(trivialIndexType);
                    }

                    for (auto val : update_bars) {
                        val.setType(trivialIndexType);
                    }

                    VPUIPRegMapped::NNDMAOp regMappedDmaTask = builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                            builderBlk.getUnknownLoc(),
                            indexType,  // ret type
                            op.input(), op.output_buff(), previousDMA, mlir::ValueRange(wait_bars),
                            mlir::ValueRange(update_bars),
                            false,  // compression
                            op.port(),
                            0  // start_after
                    );

                    previousDMA = regMappedDmaTask.getResult();

                    dma_count++;
                }
                if (foundTaskOp) {
                    taskOp->erase();
                }
                break;  // Block iterator gets invalidated after erase().
            }

            if (foundTaskOp == false)
                break;
        }  // End forever loop
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithNNDMAOp() -- end");
    }

    void replaceVPURTTaskOpWithKernelOps(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::FuncOp& funcOp,
                                         Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithKernelOps()");

        auto shave_task_count = 0;

        // Forever loop that runs until there are no more changes performed by
        //   the inner loop (so the algorithm has converged).
        for (;;) {
            bool foundTaskOp = false;

            for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
                if (!taskOp.body().getOps<VPUIP::SwKernelOp>().empty()) {
                    foundTaskOp = true;
                }
                for (auto op : taskOp.body().getOps<VPUIP::SwKernelOp>()) {
                    mlir::OpBuilder builderBlk(taskOp);

                    auto indexType = VPUIPRegMapped::IndexType::get(ctx, shave_task_count);

                    auto wait_bars = taskOp.waitBarriers();
                    auto update_bars = taskOp.updateBarriers();

                    auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

                    for (auto val : wait_bars) {
                        val.setType(trivialIndexType);
                    }

                    for (auto val : update_bars) {
                        val.setType(trivialIndexType);
                    }

                    auto sw_kernel_symbol = op.kernelFunction();

                    auto kernel_info_funcOp = moduleOp.lookupSymbol<mlir::FuncOp>(sw_kernel_symbol);

                    auto kernel_str = std::string(
                            kernel_info_funcOp->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry").getValue());

                    const auto kernel_elf = kernel_str + std::string(".3720xx") + std::string(".elf");
                    auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);

                    std::vector<uint8_t> params_vector;

                    for (auto&& kernelRun : op.body().getOps<VPUIP::SwKernelRun>()) {
                        for (auto&& operand : kernelRun.args()) {
                            auto blockArg = operand.dyn_cast_or_null<mlir::BlockArgument>();
                            if (blockArg) {
                                auto id = blockArg.getArgNumber();

                                const auto operandVal = op->getOpOperand(id).get();

                                addTensorArgToVector(params_vector, operandVal);
                            } else {
                                VPUX_THROW("Only block arguments are supported");
                            }
                        }
                        for (auto&& attr : kernelRun.attrs().getValue()) {
                            addAttrsToVector(params_vector, attr);
                        }
                    }

                    long int params_size = (long int)(params_vector.size());

                    auto kernelTextOp = builderBlk.create<VPUIPRegMapped::DeclareKernelTextOp>(
                            builderBlk.getUnknownLoc(), indexType, mlir::StringAttr::get(ctx, kernel_elf));

                    auto kernelArgsOp = builderBlk.create<VPUIPRegMapped::DeclareKernelArgsOp>(
                            builderBlk.getUnknownLoc(), indexType, mlir::StringAttr::get(ctx, kernel_elf));

                    auto kernelEntryOp = builderBlk.create<VPUIPRegMapped::DeclareKernelEntryOp>(
                            builderBlk.getUnknownLoc(), indexType, mlir::StringAttr::get(ctx, kernel_elf));

                    auto kernelRangeOp = builderBlk.create<VPUIPRegMapped::ActKernelRangeOp>(
                            builderBlk.getUnknownLoc(), indexType, kernelTextOp, kernelArgsOp, kernelEntryOp);

                    builderBlk.create<VPUIPRegMapped::ActKernelInvocationOp>(
                            builderBlk.getUnknownLoc(), indexType, mlir::ValueRange(wait_bars),
                            mlir::ValueRange(update_bars), kernelRangeOp.getResult(), 0, 0, indexType.getValue() + 1);

                    builderBlk.create<VPUIPRegMapped::KernelParamsOp>(
                            builderBlk.getUnknownLoc(), indexType, op.inputs().front(), op.output_buffs().front(),
                            mlir::StringAttr::get(ctx, kernel_str),
                            mlir::DenseIntElementsAttr::get(mlir::VectorType::get({params_size}, uint8Type),
                                                            params_vector));

                    shave_task_count++;
                }

                if (foundTaskOp) {
                    taskOp->erase();
                }

                break;  // Block iterator gets invalidated after erase().
            }

            if (foundTaskOp == false)
                break;
        }  // End forever loop
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithKernelOps() -- end");
    }

    void setBarrierIndexValues(mlir::MLIRContext* ctx, mlir::FuncOp& funcOp, Logger _log) {
        auto barrier_count = 0;

        VPUX_UNUSED(_log);

        for (auto op : funcOp.getOps<VPUIPRegMapped::ConfigureBarrierOp>()) {
            mlir::OpBuilder builderBlk(op);

            auto indexType = VPUIPRegMapped::IndexType::get(ctx, barrier_count);

            op.getOperation()->getResult(0).setType(indexType);

            barrier_count++;
        }
    }
};

class ConvertVPURTConfigureBarrierOp final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrierOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const {
        auto ctx = ConvertVPURTConfigureBarrierOp::getContext();

        auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

        mlir::Value origOpValue = origOp.getOperation()->getResult(0);

        size_t producer_count = 0;
        size_t consumer_count = 0;

        // should use VPUIPRegMapped TaskOp interface
        for (auto user : origOpValue.getUsers()) {
            if (auto dmaOp = mlir::dyn_cast<vpux::VPUIPRegMapped::NNDMAOp>(user)) {
                for (auto waitBar : dmaOp.waitBarriers()) {
                    if (origOpValue == waitBar) {
                        producer_count++;
                    }
                }
                for (auto updateBar : dmaOp.updateBarriers()) {
                    if (origOpValue == updateBar) {
                        consumer_count++;
                    }
                }
            }

            if (auto kernelInvoOp = mlir::dyn_cast<vpux::VPUIPRegMapped::ActKernelInvocationOp>(user)) {
                for (auto waitBar : kernelInvoOp.waitBarriers()) {
                    if (origOpValue == waitBar) {
                        producer_count++;
                    }
                }
                for (auto updateBar : kernelInvoOp.updateBarriers()) {
                    if (origOpValue == updateBar) {
                        consumer_count++;
                    }
                }
            }
        }

        mlir::IntegerType uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);

        rewriter.replaceOpWithNewOp<VPUIPRegMapped::ConfigureBarrierOp>(
                origOp,
                trivialIndexType,                                   // setup all barriers with the trivial index (0)
                origOp.id(),                                        // real_id
                -1,                                                 // int32_t next_same_id()
                mlir::IntegerAttr::get(uint8Type, producer_count),  // origOp.producer_countAttr(),
                mlir::IntegerAttr::get(uint8Type, consumer_count)   // origOp.consumer_countAttr(),
        );
        barrier_count++;
        return mlir::success();
    }

private:
    Logger _log;
    mutable int barrier_count = 0;
};

void ConvertVPUIP2VPUIPRegMappedPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();
    for (auto op : moduleOp.getOps<mlir::FuncOp>()) {
        funcOp = op;
        break;
    }

    _log.info("funcOp = {0}", funcOp);

    replaceVPURTTaskOpWithNNDMAOp(ctx, funcOp, _log);

    _log.info("funcOp after replacing NNDMA Ops = {0}", funcOp);

    replaceVPURTTaskOpWithKernelOps(ctx, moduleOp, funcOp, _log);

    _log.info("funcOp after replacing ActKernel Ops = {0}", funcOp);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<VPURT::DeclareBufferOp>();

    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ConvertVPURTConfigureBarrierOp>(ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
    }

    _log.info("funcOp after replacing Barrier Ops = {0}", funcOp);

    setBarrierIndexValues(ctx, funcOp, _log);

    _log.info("funcOp after setting Barrier indexes = {0}", funcOp);
}

}  // namespace

//
// createConvertVPUIP2VPUIPRegMappedPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUIPRegMappedPass(Logger log) {
    return std::make_unique<ConvertVPUIP2VPUIPRegMappedPass>(log);
}
