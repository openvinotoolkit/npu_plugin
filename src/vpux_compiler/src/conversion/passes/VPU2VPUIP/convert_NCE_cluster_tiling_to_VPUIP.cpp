//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

namespace {

using GroupedAllocMap = DenseMap<mlir::Operation*, SmallVector<mlir::Operation*>>;

//
// NCEClusterTilingRewriter
//

class NCEClusterTilingRewriter final : public mlir::OpConversionPattern<VPU::NCEClusterTilingOp> {
public:
    NCEClusterTilingRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEClusterTilingOp>(converter, ctx), _ctx(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    mlir::MLIRContext* _ctx;
    Logger _log;
};

mlir::LogicalResult NCEClusterTilingRewriter::matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    VPUX_THROW_UNLESS(origOp.results().size() == 1,
                      "Currently only single output NCEClusterTiling operation is supported, got {0} outputs",
                      origOp.results().size());

    SmallVector<mlir::Value> inputOperands = newArgs.operands();
    SmallVector<mlir::Value> origOutputBuffers;

    const auto clusterTilingBufferType = typeConverter->convertType(origOp.results()[0].getType());

    auto outputDataDistType = clusterTilingBufferType.dyn_cast<VPUIP::DistributedBufferType>();
    if (auto sparseType = clusterTilingBufferType.dyn_cast<VPUIP::SparseBufferType>()) {
        outputDataDistType = sparseType.getData().dyn_cast<VPUIP::DistributedBufferType>();
    }

    auto& origOpBodyBlock = origOp.body().front();

    rewriter.setInsertionPoint(origOp);

    // The goal is to move the allocation operations outside the NCEClusterTiling body.
    // In case the resulting type of NCEClusterTiling is a distributed type, these allocations
    // should also produce distributed types now that they are outside.
    // This function will ensure the correct buffer type is allocated.
    const auto createNewAllocOp = [&](mlir::Operation* op) {
        const auto outputType = op->getResult(0).getType();
        if (outputDataDistType != nullptr) {
            const auto ndOutputType = outputType.cast<vpux::NDTypeInterface>();
            const auto layout = mlir::AffineMapAttr::get(ndOutputType.getDimsOrder().toAffineMap(_ctx));
            auto distributedBufferType = VPUIP::DistributedBufferType::get(
                    _ctx, ndOutputType.getShape().raw(), ndOutputType.getElementType(), layout,
                    ndOutputType.getMemSpace(), outputDataDistType.getDistribution());
            return rewriter.create<VPURT::AllocDistributed>(op->getLoc(), distributedBufferType, nullptr, nullptr)
                    .getOperation();
        }
        const auto memrefOutputType = outputType.cast<mlir::MemRefType>();
        return rewriter.create<mlir::memref::AllocOp>(op->getLoc(), memrefOutputType).getOperation();
    };
    // checks whether given operation is a direct Inner or a level 2 or more inner
    const auto isLevelOneOp = [&origOpBodyBlock, this](mlir::Operation* op) -> bool {
        return op->getBlock() == &origOpBodyBlock;
    };

    const auto isInnerOp = [isLevelOneOp](mlir::Operation* op) -> bool {
        return !mlir::isa<VPU::YieldOp, VPUIP::UngroupSparseBufferOp, VPUIP::GroupSparseBufferOp,
                          mlir::UnrealizedConversionCastOp>(op) &&
               !isBufAllocOp(op) && isLevelOneOp(op);
    };

    const auto skipUserCast = [](mlir::Value operand) -> mlir::Value {
        if (!operand.hasOneUse()) {
            return operand;
        }
        auto userOp = *operand.getUsers().begin();
        if (mlir::isa<mlir::UnrealizedConversionCastOp>(userOp)) {
            return userOp->getResult(0);
        }
        return operand;
    };

    const auto skipProducerCast = [](mlir::Value operand) -> mlir::Value {
        if (auto castOp = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
            return castOp.getOperand(0);
        }
        return operand;
    };

    // Clone the allocation and grouping / ungrouping operations outside of the cluster tiling region
    DenseMap<mlir::Value, mlir::Value> innerOuterOpValueMapping;
    SmallVector<mlir::Operation*> newAllocOps;
    SmallVector<mlir::Operation*> newGroupOps;
    origOpBodyBlock.walk([&](mlir::Operation* op) {
        if (vpux::isBufAllocOp(op)) {
            _log.nest().trace("Cloning allocation op '{0}' at '{1}", op->getName(), op->getLoc());
            auto newAllocOp = createNewAllocOp(op);
            newAllocOps.push_back(newAllocOp);
            innerOuterOpValueMapping[op->getResult(0)] = newAllocOp->getResult(0);

        } else if (mlir::isa<VPUIP::UngroupSparseBufferOp>(op)) {
            _log.nest().trace("Cloning ungrouping op '{0}' at '{1}", op->getName(), op->getLoc());

            auto operand = op->getOperand(0);
            auto blockArg = skipProducerCast(operand).dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_UNLESS(blockArg != nullptr, "Expected operand of ungroup op to be a block argument");
            auto outerArg = origOp.getOperand(blockArg.getArgNumber());
            outerArg = skipProducerCast(outerArg);

            auto outerType = typeConverter->convertType(outerArg.getType());
            auto outerValueCast = rewriter.create<mlir::UnrealizedConversionCastOp>(
                    op->getLoc(), mlir::TypeRange{outerType}, mlir::ValueRange{outerArg});

            mlir::BlockAndValueMapping mapper;
            mapper.map(operand, outerValueCast.getResult(0));
            auto newOp = rewriter.clone(*op, mapper);
            vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

            for (auto i : irange(op->getNumResults())) {
                innerOuterOpValueMapping[op->getResult(i)] = newOp->getResult(i);
            }

        } else if (mlir::isa<VPUIP::GroupSparseBufferOp>(op)) {
            if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
                    return vpux::isBufAllocOp(operand.getDefiningOp());
                })) {
                return;
            }
            _log.nest().trace("Cloning grouping op '{0}' at '{1}", op->getName(), op->getLoc());

            mlir::BlockAndValueMapping mapper;
            for (auto operand : op->getOperands()) {
                mapper.map(operand, innerOuterOpValueMapping[operand]);
            }
            auto newOp = rewriter.clone(*op, mapper);
            vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
            newGroupOps.push_back(newOp);
            innerOuterOpValueMapping[op->getResult(0)] = newOp->getResult(0);
        }
    });

    SmallVector<mlir::Value> newOutputOperands;
    auto& newOutputOps = !newGroupOps.empty() ? newGroupOps : newAllocOps;
    for (auto op : newOutputOps) {
        newOutputOperands.push_back(op->getResult(0));
    }

    // Create the vector of operands for the new bufferized cluster tiling op starting with the input operands
    // Additionally, get the mapping from the input operands index to the new indices of the operands
    SmallVector<mlir::Value> newOperands;
    DenseMap<int64_t, SmallVector<int64_t>> operandsIdxMapping;
    for (auto p : inputOperands | indexed) {
        auto operandIdx = p.index();
        auto operand = p.value();
        if (!operand.getType().isa<VPUIP::SparseBufferType>()) {
            newOperands.push_back(operand);
            operandsIdxMapping[operandIdx].push_back(newOperands.size() - 1);
            continue;
        }

        mlir::Value origOperand = origOpBodyBlock.getArguments()[operandIdx];
        origOperand = skipUserCast(origOperand);

        if (llvm::any_of(origOperand.getUsers(), [](mlir::Operation* userOp) {
                return !mlir::isa<VPUIP::UngroupSparseBufferOp>(userOp);
            })) {
            newOperands.push_back(operand);
            operandsIdxMapping[operandIdx].push_back(newOperands.size() - 1);
            continue;
        }

        for (auto userOp : origOperand.getUsers()) {
            for (auto result : userOp->getResults()) {
                auto it = innerOuterOpValueMapping.find(result);
                newOperands.push_back(it->second);
                operandsIdxMapping[operandIdx].push_back(newOperands.size() - 1);
            }
        }
    }

    // Add the output operands to the new vector of operands
    // Maintain a mapping between the outer output operand and its index in the new operands vector
    DenseMap<mlir::Value, int64_t> outputOperandsMapping;
    for (auto outputOperand : newOutputOperands) {
        newOperands.push_back(outputOperand);
        outputOperandsMapping[outputOperand] = newOperands.size() - 1;
    }

    // Get the mapping from the inner op operands to the new block operands
    DenseMap<mlir::Value, int64_t> innerOpOperandMapping;
    origOpBodyBlock.walk([&](mlir::Operation* op) {
        if (!isInnerOp(op)) {
            return;
        }

        for (auto p : op->getOperands() | indexed) {
            auto operand = p.value();
            if (auto blockArg = skipProducerCast(operand).dyn_cast<mlir::BlockArgument>()) {
                innerOpOperandMapping[operand] = operandsIdxMapping[blockArg.getArgNumber()].front();
            } else if (auto ungroupOp = operand.getDefiningOp<VPUIP::UngroupSparseBufferOp>()) {
                auto ungroupInput = skipProducerCast(ungroupOp.input());
                auto blockArg = ungroupInput.dyn_cast<mlir::BlockArgument>();
                VPUX_THROW_UNLESS(blockArg != nullptr, "Unexpected operand for ungrouping operation");
                auto resultIt = llvm::find(ungroupOp->getResults(), operand);
                auto resultIdx = std::distance(ungroupOp->getResults().begin(), resultIt);
                innerOpOperandMapping[operand] = operandsIdxMapping[blockArg.getArgNumber()][resultIdx];
            } else {
                auto outerOperand = innerOuterOpValueMapping[operand];
                innerOpOperandMapping[operand] = outputOperandsMapping[outerOperand];
            }
        }
    });

    // Compute the new resulting types of the bufferized cluster tiling op
    SmallVector<mlir::Type> newClusterTilingResultTypes;
    for (auto outputOperand : newOutputOperands) {
        newClusterTilingResultTypes.push_back(outputOperand.getType());
    }

    // Create a VPUIP::NCEClusterTiling operation whose body contains only the inner operation
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location /*loc*/, mlir::ValueRange newArgs) {
        for (auto& op : origOpBodyBlock.getOperations()) {
            if (!isInnerOp(&op)) {
                continue;
            }

            _log.nest().trace("Cloning op '{0}' at '{1}", op.getName(), op.getLoc());

            mlir::BlockAndValueMapping mapper;
            for (auto operand : op.getOperands()) {
                auto it = innerOpOperandMapping.find(operand);
                VPUX_THROW_UNLESS(it != innerOpOperandMapping.end(), "No mapping found for operand of inner op");
                mapper.map(operand, newArgs[it->second]);
            }
            builder.clone(op, mapper);
        }
    };
    auto clusterTilingOp = rewriter.create<VPUIP::NCEClusterTilingOp>(origOp->getLoc(), newClusterTilingResultTypes,
                                                                      newOperands, bodyBuilder);

    auto outputResults = clusterTilingOp.results();

    // Add an output grouping operation if the original cluster tiling operation returned a grouped result
    mlir::Operation* outputGroupingOp = nullptr;
    origOpBodyBlock.walk([&](VPUIP::GroupSparseBufferOp groupOp) {
        if (llvm::none_of(groupOp->getOperands(), [](mlir::Value operand) {
                return isBufAllocOp(operand.getDefiningOp());
            })) {
            VPUX_THROW_UNLESS(outputGroupingOp == nullptr, "Multiple grouping operations are not allowed");
            outputGroupingOp = groupOp;
        }
    });
    if (outputGroupingOp != nullptr) {
        auto groupOp = rewriter.create<VPUIP::GroupSparseBufferOp>(outputGroupingOp->getLoc(), outputResults,
                                                                   outputGroupingOp->getAttrs());
        outputResults = groupOp->getResults();
    }

    rewriter.replaceOp(origOp, outputResults);

    return mlir::success();
}

//
// ConvertNCEClusterTilingToVPUIPPass
//

class ConvertNCEClusterTilingToVPUIPPass final :
        public ConvertNCEClusterTilingToVPUIPBase<ConvertNCEClusterTilingToVPUIPPass> {
public:
    explicit ConvertNCEClusterTilingToVPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertNCEClusterTilingToVPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    vpux::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addIllegalOp<VPU::NCEClusterTilingOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<NCEClusterTilingRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNCEClusterTilingToVPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertNCEClusterTilingToVPUIPPass(Logger log) {
    return std::make_unique<ConvertNCEClusterTilingToVPUIPPass>(log);
}
