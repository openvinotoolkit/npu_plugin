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
    void appendValueToVector(SmallVector<uint8_t>& vec, const T& anyValue) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&anyValue), sizeof(anyValue));
        vec.insert(vec.end(), valueAsArray.begin(), valueAsArray.end());
    }

    void addBasicAttrToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
        if (auto val = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
            appendValueToVector(vec, val.getValue().getSExtValue());
        } else if (auto val = attr.dyn_cast_or_null<mlir::FloatAttr>()) {
            appendValueToVector(vec, static_cast<float>(val.getValue().convertToDouble()));
        } else {
            VPUX_THROW("Act Shave Invocation: cannot store arg of type {0}", attr.getType());
        }
    }

    void addAttrsToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
        if (auto arr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
            auto vals = arr.getValue();
            for (auto val : vals) {
                addBasicAttrToVector(vec, val);
            }
        } else {
            addBasicAttrToVector(vec, attr);
        }
    }

    void addTensorArgToVector(SmallVector<uint8_t>& vec, mlir::Value value) {
        sw_params::MemRefData memrefData{};

        const auto shape = getShape(value);
        memrefData.numDims = checked_cast<uint32_t>(shape.size());

        // order
        const auto inOrder = DimsOrder::fromValue(value);
        const auto memShape = inOrder.toMemoryOrder(shape);
        memrefData.dimsOrder = inOrder.invertedCode();

        memrefData.dataType = 0;  // TODO: extract data from SWTaskOp E#54004
        memrefData.location = sw_params::NN_CMX;

        appendValueToVector(vec, memrefData);
    }

    SmallVector<uint8_t> createKernelParams(VPUIP::SwKernelOp swKernelOp) {
        SmallVector<uint8_t> paramsVector;

        const auto insSize = swKernelOp.inputs().size();
        const auto outsSize = swKernelOp.results().size();

        const auto kernelOpArgsCount = insSize + outsSize;

        for (auto&& kernelRun : swKernelOp.body().getOps<VPUIP::SwKernelRun>()) {
            for (auto&& operand : kernelRun.args()) {
                auto blockArg = operand.dyn_cast_or_null<mlir::BlockArgument>();
                if (blockArg) {
                    auto id = blockArg.getArgNumber();
                    VPUX_THROW_UNLESS(id < kernelOpArgsCount,
                                      "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", id,
                                      kernelOpArgsCount);

                    auto blockArgType = blockArg.getType();
                    auto blockArgNdTypeIf = blockArgType.cast<vpux::NDTypeInterface>();
                    auto ioType = id < insSize ? swKernelOp.inputs()[id].getType()
                                               : swKernelOp.output_buffs()[insSize - id].getType();
                    auto ioNdTypeIf = ioType.cast<vpux::NDTypeInterface>();
                    VPUX_THROW_UNLESS(blockArgNdTypeIf != nullptr || ioNdTypeIf != nullptr,
                                      "createKernelParams: sw kernel I/O does not implement NDTypeInterface");
                    VPUX_THROW_UNLESS(blockArgType == ioType,
                                      "createKernelParams: types of sw kernel I/O do not match");
                    VPUX_THROW_UNLESS(blockArgNdTypeIf.getShape() == ioNdTypeIf.getShape(),
                                      "createKernelParams: shapes of I/O do not match");

                    const auto operandVal = swKernelOp->getOpOperand(id).get();
                    addTensorArgToVector(paramsVector, operandVal);
                } else {
                    VPUX_THROW("Only block arguments are supported");
                }
            }
            if (kernelRun.attrs().hasValue()) {
                const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
                const auto& attrs = arrayAttrs.getValue();
                for (const auto& attr : attrs) {
                    addAttrsToVector(paramsVector, attr);
                }
            }
        }

        return paramsVector;
    }

    void replaceVPURTTaskOpWithNNDMAOp(mlir::MLIRContext* ctx, mlir::FuncOp& funcOp, Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithNNDMAOp()");

        auto dma_count = 0;

        mlir::Value previousDMA;

        for (auto taskOp : llvm::make_early_inc_range(funcOp.body().getOps<VPURT::TaskOp>())) {
            bool found = false;

            for (auto op : llvm::make_early_inc_range(taskOp.body().getOps<VPUIP::NNDMAOp>())) {
                found = true;
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

            if (found) {
                taskOp->erase();
            }
        }

        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithNNDMAOp() -- end");
    }

    void replaceVPURTTaskOpWithKernelOps(mlir::MLIRContext* ctx, mlir::ModuleOp moduleOp, mlir::FuncOp funcOp,
                                         Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithKernelOps()");

        auto shave_task_count = 0;

        // Forever loop that runs until there are no more changes performed by
        //   the inner loop (so the algorithm has converged).

        for (auto taskOp : llvm::make_early_inc_range(funcOp.body().getOps<VPURT::TaskOp>())) {
            bool found = false;

            for (auto op : llvm::make_early_inc_range(taskOp.body().getOps<VPUIP::SwKernelOp>())) {
                found = true;
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

                auto kernel_elf =
                        std::string(kernel_info_funcOp->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry").getValue());

                auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);

                SmallVector<uint8_t> paramsVector = createKernelParams(op);

                long int paramsSize = (long int)(paramsVector.size());

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
                        mlir::ValueRange(update_bars), kernelRangeOp.getResult(), /* tile= */ 0,
                        /* start_after= */ 0, /* clean_after= */ 0);

                builderBlk.create<VPUIPRegMapped::KernelParamsOp>(
                        builderBlk.getUnknownLoc(), indexType, op.inputs(), op.output_buffs(),
                        mlir::StringAttr::get(ctx, kernel_elf),
                        mlir::DenseIntElementsAttr::get(mlir::VectorType::get({paramsSize}, uint8Type), paramsVector));

                shave_task_count++;
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void replaceNCEClusterTaskOpWithDPUOps(mlir::MLIRContext* ctx, mlir::FuncOp funcOp, Logger _log) {
        int variant_task_count = 0;
        int invariant_task_count = 0;

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getOps<VPURT::TaskOp>())) {
            bool found = false;

            _log.info("replaceNCEClusterTaskOpWithDPUOps(): taskOp = {0}", taskOp);

            for (auto op : llvm::make_early_inc_range(taskOp.body().getOps<VPUIP::NCEClusterTaskOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                if (op.input().getType().isa<VPUIP::SparseBufferType>() ||
                    op.weights().getType().isa<VPUIP::SparseBufferType>() ||
                    op.parent_input().getType().isa<VPUIP::DistributedBufferType>() ||
                    op.parent_output().getType().isa<VPUIP::DistributedBufferType>() ||
                    op.output_buff().getType().isa<VPUIP::DistributedBufferType>()) {
                    _log.error("VPUIPRegMapped conversion not supported for multiCluster and sparsity usecases");
                    signalPassFailure();
                    break;
                }

                auto wait_barriers = taskOp.waitBarriers();
                auto update_barriers = taskOp.updateBarriers();

                auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_barriers) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_barriers) {
                    val.setType(trivialIndexType);
                }

                const auto& dpuTasks = op.variants().getOps<VPUIP::DPUTaskOp>();
                VPUX_THROW_UNLESS(!dpuTasks.empty(), "Encountered op {} with empty dpu list", op);
                const auto& differentMPEModes = std::adjacent_find(dpuTasks.begin(), dpuTasks.end(),
                                                                   [](VPUIP::DPUTaskOp lhs, VPUIP::DPUTaskOp rhs) {
                                                                       return lhs.mpe_mode() != rhs.mpe_mode();
                                                                   });
                if (differentMPEModes != dpuTasks.end()) {
                    VPUIP::DPUTaskOp lhs = *differentMPEModes;
                    VPUIP::DPUTaskOp rhs = *std::next(differentMPEModes);
                    VPUX_THROW("Found dpu tasks {} and {} inside of {} which has different MPE modes {} and {} "
                               "accordingly, but only uniform MPE mode is supported by ELF",
                               lhs, rhs, op, lhs.mpe_mode(), rhs.mpe_mode());
                }

                VPUIP::DPUTaskOp first = *(dpuTasks.begin());
                auto mpe_freq_mode = VPU::MPEModeAttr::get(ctx, first.mpe_mode());
                auto invariantIndex = VPUIPRegMapped::IndexType::get(ctx, invariant_task_count);
                auto startAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);
                auto cleanAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);

                auto inv = builderBlk.create<VPUIPRegMapped::DPUInvariantOp>(
                        builderBlk.getUnknownLoc(), invariantIndex, op.input(), op.weights(), op.weight_table(),
                        op.parent_input(), op.parent_output(), op.output_buff(), op.profiling_data(),
                        op.task_typeAttr(), mpe_freq_mode, op.kernel_sizeAttr(), op.kernel_stridesAttr(),
                        op.kernel_paddingAttr(), op.activation_window_channel_lengthAttr(), op.is_continuedAttr(),
                        op.cm_sp_patternAttr(), op.input_channels_compressionAttr(), op.is_segmentedAttr(),
                        op.out_channel_offsetAttr(), wait_barriers, update_barriers, startAfterAttr, cleanAfterAttr);

                invariant_task_count++;

                for (auto dpuTaskOp : op.variants().getOps<VPUIP::DPUTaskOp>()) {
                    auto variantIndex = VPUIPRegMapped::IndexType::get(ctx, variant_task_count);
                    builderBlk.create<VPUIPRegMapped::DPUVariantOp>(
                            builderBlk.getUnknownLoc(), variantIndex, inv.getResult(), dpuTaskOp.startAttr(),
                            dpuTaskOp.endAttr(), dpuTaskOp.padAttr(), dpuTaskOp.mpe_modeAttr(),
                            dpuTaskOp.cluster_idAttr());
                    variant_task_count++;
                }

                if (op.ppe().hasOneBlock()) {
                    mlir::BlockAndValueMapping mapper;
                    op.ppe().cloneInto(&inv.ppe(), mapper);
                }
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void setBarrierIndexValues(mlir::MLIRContext* ctx, mlir::FuncOp& funcOp, Logger _log) {
        auto barrier_count = 0;

        VPUX_UNUSED(_log);

        for (auto op : funcOp.getOps<VPUIPRegMapped::ConfigureBarrierOp>()) {
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

        mlir::Value origOpResult = origOp.getResult();

        size_t producer_count = 0;
        size_t consumer_count = 0;

        // should use VPUIPRegMapped TaskOp interface
        for (auto user : origOpResult.getUsers()) {
            if (auto taskOp = mlir::dyn_cast<vpux::VPUIPRegMapped::ExecutableTaskOpInterface>(user)) {
                for (auto waitBar : taskOp.waitBarriers()) {
                    if (origOpResult == waitBar) {
                        consumer_count++;
                    }
                }

                for (auto updateBar : taskOp.updateBarriers()) {
                    if (origOpResult == updateBar) {
                        producer_count++;
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

    replaceNCEClusterTaskOpWithDPUOps(ctx, funcOp, _log);

    _log.info("funcOp after replacing DPU Ops = {0}", funcOp);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<VPURT::DeclareBufferOp>();
    target.addLegalOp<VPUIP::PPETaskOp>();

    mlir::RewritePatternSet patterns(ctx);

    patterns.add<ConvertVPURTConfigureBarrierOp>(ctx, _log);

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
