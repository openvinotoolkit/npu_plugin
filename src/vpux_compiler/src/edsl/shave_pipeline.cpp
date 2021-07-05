//
// Copyright 2021 Intel Corporation.
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

#include <llvm/ADT/SetVector.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

#ifdef ENABLE_PLAIDML
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/transforms/cache.h"
#include "pmlc/dialect/pxa/transforms/normalize.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/pxa/transforms/tile_accumulate.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/util.h"
#endif

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/edsl/autotile.hpp"
#include "vpux/compiler/edsl/passes.hpp"
#include "vpux/compiler/edsl/utils.hpp"

using namespace mlir;          // NOLINT
using namespace mlir::memref;  // NOLINT
using namespace vpux::VPUIP;   // NOLINT

#ifdef ENABLE_PLAIDML
namespace pxa = pmlc::dialect::pxa;
#endif

namespace vpux {
namespace edsl {

#ifdef ENABLE_PLAIDML

struct Initializer {
    Value source;
    Value result;
    Attribute value;
};

// TODO: move to upstream
static AffineMap getFlatAffineMap(MLIRContext* context, const pxa::RelativeAccessPattern& rap,
                                  ArrayRef<BlockArgument> idxs) {
    auto flatOuter = rap.flatOuter();
    assert(flatOuter.hasValue() && "flatOuter failed");
    auto expr = getAffineConstantExpr(flatOuter->offset, context);
    for (auto idx : llvm::enumerate(idxs)) {
        auto it = flatOuter->strides.find(idx.value());
        if (it == flatOuter->strides.end()) {
            continue;
        }
        auto factor = it->second;
        expr = expr + factor * getAffineDimExpr(idx.index(), context);
    }
    return AffineMap::get(idxs.size(), 0, expr);
}

static EdslDMADirection getDirection(const pxa::CachePlan::Entry& entry) {
    if (entry.copyInto && entry.copyFrom) {
        return EdslDMADirection::INOUT;
    }
    if (entry.copyInto) {
        return EdslDMADirection::IN;
    }
    if (entry.copyFrom) {
        return EdslDMADirection::OUT;
    }
    return EdslDMADirection::NONE;
}

struct KernelFuncHelper {
    Region& body;
    unsigned numOuterArgs;
    unsigned numMiddleArgs;

    KernelFuncHelper(FuncOp funcOp, unsigned numOuterArgs, unsigned numMiddleArgs)
            : body(funcOp.getBody()), numOuterArgs(numOuterArgs), numMiddleArgs(numMiddleArgs) {
    }

    ArrayRef<BlockArgument> getOuterArgs() {
        auto begin = body.args_begin();
        auto end = std::next(begin, numOuterArgs);
        return {begin, end};
    }

    ArrayRef<BlockArgument> getMiddleArgs() {
        auto begin = std::next(body.args_begin(), numOuterArgs);
        auto end = std::next(begin, numMiddleArgs);
        return {begin, end};
    }

    ArrayRef<BlockArgument> getInnerArgs() {
        auto begin = std::next(body.args_begin(), numOuterArgs + numMiddleArgs);
        return {begin, body.args_end()};
    }
};

static LogicalResult sinkOperationsIntoKernel(FuncOp funcOp) {
    auto& body = funcOp.getBody();

    // Identify uses from values defined outside of the scope of the source
    // operation.
    llvm::SetVector<Value> sinkCandidates;
    getUsedValuesDefinedAbove(body, sinkCandidates);

    llvm::SetVector<Value> sunkValues;
    llvm::SetVector<Operation*> sunkOperations;
    for (Value operand : sinkCandidates) {
        Operation* operandOp = operand.getDefiningOp();
        if (!operandOp) {
            continue;
        }
        // Only sink operations that do not create new sinkCandidates.
        if (!llvm::all_of(operandOp->getOperands(), [&sinkCandidates](Value value) {
                return sinkCandidates.count(value);
            })) {
            continue;
        }
        sunkValues.insert(operand);
        sunkOperations.insert(operandOp);
    }

    // Insert operations so that the defs get cloned before uses.
    BlockAndValueMapping map;
    OpBuilder builder(body);
    DenseSet<Operation*> processed;
    SmallVector<Operation*, 2> clonedOps;
    while (processed.size() != sunkOperations.size()) {
        auto startSize = processed.size();
        for (Operation* sunkOperation : sunkOperations) {
            if (processed.count(sunkOperation)) {
                continue;
            }

            // Operation can't be cloned yet if any of its operands are also being
            // sunk, but isn't cloned yet.
            if (llvm::any_of(sunkOperation->getOperands(), [&sunkValues, &map](Value value) {
                    return sunkValues.count(value) && !map.lookupOrNull(value);
                })) {
                continue;
            }

            Operation* clonedOp = builder.clone(*sunkOperation, map);
            // Only replace uses within the func op.
            for (auto result : llvm::enumerate(sunkOperation->getResults())) {
                auto replacement = clonedOp->getResult(result.index());
                for (auto& use : llvm::make_early_inc_range(result.value().getUses())) {
                    if (use.getOwner()->getParentOfType<FuncOp>() == funcOp) {
                        use.set(replacement);
                    }
                }
            }
            processed.insert(sunkOperation);
        }
        if (startSize == processed.size()) {
            return funcOp.emitError("found illegal cyclic dependency between operations while sinking");
        }
    }
    return success();
}

static Optional<Initializer> detectScalarInitializer(Operation* op, Value result) {
    Value source;
    FloatAttr floatAttr;
    IntegerAttr intAttr;
    Builder builder(op->getContext());
    if (pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Constant(&floatAttr),
                           pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Constant(), m_Capture(&source)))
                .match(op) ||
        pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Constant(&floatAttr), m_Capture(&source)).match(op) ||
        pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Op<FPExtOp>(m_Constant(&floatAttr)), m_Capture(&source))
                .match(op) ||
        pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Op<FPTruncOp>(m_Constant(&floatAttr)), m_Capture(&source))
                .match(op)) {
        return Initializer{source, result, floatAttr};
    }
    if (pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Constant(&intAttr), m_Capture(&source)).match(op)) {
        return Initializer{source, result, intAttr};
    }
    if (pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Op<SIToFPOp>(m_Constant(&intAttr)), m_Capture(&source)).match(op) ||
        pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Op<UIToFPOp>(m_Constant(&intAttr)), m_Capture(&source)).match(op)) {
        return Initializer{source, result, builder.getF64FloatAttr(intAttr.getInt())};
    }
    if (pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Op<FPToSIOp>(m_Constant(&floatAttr)), m_Capture(&source))
                .match(op) ||
        pxa::m_PxaReduceOp(AtomicRMWKind::assign, m_Op<FPToUIOp>(m_Constant(&floatAttr)), m_Capture(&source))
                .match(op)) {
        return Initializer{source, result, builder.getI64IntegerAttr(floatAttr.getValueAsDouble())};
    }
    return None;
}

static Optional<Initializer> detectLoopInitializer(AffineParallelOp band) {
    if (!band.getNumResults()) {
        return None;
    }
    auto yield = band.getBody()->getTerminator();
    return detectScalarInitializer(yield->getOperand(0).getDefiningOp(), band.getResult(0));
}

static FrozenRewritePatternSet collectCanonPatterns(MLIRContext* context) {
    RewritePatternSet patterns(context);
    for (auto* op : context->getRegisteredOperations()) {
        op->getCanonicalizationPatterns(patterns, context);
    }
    return patterns;
}

struct ShavePipelineImpl {
    FuncOp program;
    ModuleOp module;
    MLIRContext* context;
    OpBuilder moduleBuilder;
    DenseMap<Value, int64_t> bufferIds;
    DenseMap<Value, Initializer> initializers;
    IndexType indexType;
    FrozenRewritePatternSet canonPatterns;
    AutoTileParams params;

    ShavePipelineImpl(FuncOp program, ModuleOp module)
            : program(program),
              module(module),
              context(module.getContext()),
              moduleBuilder(OpBuilder::atBlockTerminator(module.getBody())),
              indexType(moduleBuilder.getIndexType()),
              canonPatterns(collectCanonPatterns(module.getContext())),
              params{
                      60000,                                 // totalBuffer
                      3,                                     // minCount
                      std::numeric_limits<uint32_t>::max(),  // maxCount
                      1024,                                  // minInnerBuffer
                      64,                                    // CacheWidth
                      16,                                    // vectorWidth
                      "outermost",                           // processingTags
                      "outer",                               // outerTags
                      "inner",                               // innerTags
                      false,                                 // outputIndicesOnly
                      false,                                 // accIndicesOnly
                      true,                                  // noNegativeIndex
              } {
        assignBufferIdentifiers(program);
    }

    LogicalResult runOnProgram(StringRef funcName) {
        for (auto& op : program.getOps()) {
            if (!op.getNumResults()) {
                continue;
            }
            if (Optional<Initializer> init = detectScalarInitializer(&op, op.getResult(0))) {
                if (failed(processInitializer(init.getPointer()))) {
                    return failure();
                }
            }
        }
        SmallVector<AffineParallelOp, 8> toRemove;
        for (auto ivp : llvm::enumerate(program.getOps<AffineParallelOp>())) {
            AffineParallelOp outer = ivp.value();
            if (failed(runOnOuterBand(funcName, ivp.index(), outer))) {
                return failure();
            }
            toRemove.push_back(outer);
        }
        for (AffineParallelOp op : toRemove) {
            op.erase();
        }
        return success();
    }

    LogicalResult processInitializer(Initializer* init) {
        Optional<int64_t> bufferId = getBufferIdentifier(init->source);
        if (!bufferId) {
            init->source.getDefiningOp()->emitOpError("Could not resolve buffer ID for init");
            return failure();
        }
        init->result.replaceAllUsesWith(init->source);
        bufferIds[init->result] = *bufferId;
        initializers[init->source] = *init;
        return success();
    }

    LogicalResult runOnOuterBand(StringRef funcName, size_t kernelId, AffineParallelOp outer) {
        if (Optional<Initializer> init = detectLoopInitializer(outer)) {
            if (failed(processInitializer(init.getPointer()))) {
                return failure();
            }
            return success();
        }

        auto innerTile = computeBestTile(outer, params);
        if (innerTile.empty()) {
            IVLOG(1, "No tile found");
            innerTile = *outer.getConstantRanges();
        }

        IVLOG(3, "tile: " << innerTile);

        // Perform the primary innermost tiling for computation.
        auto inner = pxa::performTiling(outer, innerTile);

        // Tile over accumulations.
        auto middle = pxa::tileAccumulations(outer, /*skipTrivial=*/false);

        // Collect the outer indexes and ranges.
        SmallVector<Type, 4> innerOperandTypes;
        llvm::SetVector<BlockArgument> outerIdxs;
        llvm::SetVector<BlockArgument> outerAndMiddleIdxs;
        SmallVector<int64_t, 8> outerRanges;
        auto maybeOuterRanges = outer.getConstantRanges();
        if (!maybeOuterRanges) {
            return outer.emitOpError("Outer ranges must be constant");
        }

        for (auto item : llvm::zip(*maybeOuterRanges, outer.getSteps(), outer.getIVs())) {
            int64_t range;
            int64_t step;
            BlockArgument idx;
            std::tie(range, step, idx) = item;
            if (range == step) {
                continue;
            }
            outerIdxs.insert(idx);
            outerAndMiddleIdxs.insert(idx);
            outerRanges.push_back(llvm::divideCeil(range, step));
            innerOperandTypes.push_back(indexType);
        }

        // Collect the middle indexes and ranges.
        SmallVector<int64_t, 8> middleRanges;
        auto maybeMiddleRanges = middle.getConstantRanges();
        if (!maybeMiddleRanges) {
            return middle.emitOpError("Middle ranges must be constant");
        }

        for (auto item : llvm::zip(*maybeMiddleRanges, middle.getSteps(), middle.getIVs())) {
            int64_t range;
            int64_t step;
            BlockArgument idx;
            std::tie(range, step, idx) = item;
            if (range == step) {
                continue;
            }
            outerAndMiddleIdxs.insert(idx);
            middleRanges.push_back(llvm::divideCeil(range, step));
            innerOperandTypes.push_back(indexType);
        }

        // Perform the caching transformations.
        pxa::CachePlan plan(outer, middle, /*wholeBlock=*/true);
        inner.walk([&](pxa::PxaLoadOp load) {
            plan.addLoad(load);
        });
        inner.walk([&](pxa::PxaReduceOp reduce) {
            plan.addReduce(reduce);
        });
        plan.execute();

        SmallVector<Value, 4> inputs;
        SmallVector<Value, 4> outputs;
        SmallVector<Type, 4> outputTypes;
        SmallVector<Value, 4> innerOperands;
        SmallVector<VPUIP_EdslDMADesc, 4> dmaDescs;
        SmallVector<int64_t, 2> outputBufferIds;
        SmallVector<Attribute, 4> inits;
        OpBuilder builder(outer);

        // Some loads are not cached, we make a DMA descriptor for the whole tensor.
        // Runtime will finally determine if it is real DMA transfer or direct DDR
        // access.
        inner.walk([&](LoadOp load) {
            auto memref = load.memref();
            auto baseMap = AffineMap::getConstantMap(0, context);
            auto init = initializers.lookup(memref);
            if (init.value) {
                inits.push_back(init.value);
            } else {
                inits.push_back(builder.getUnitAttr());
            }
            auto dma = VPUIP_EdslDMADesc::get(AffineMapAttr::get(baseMap),
                                              EdslDMAStageAttr::get(context, EdslDMAStage::ALL),
                                              EdslDMADirectionAttr::get(context, EdslDMADirection::IN), context);
            dmaDescs.push_back(dma);
            inputs.push_back(memref);
            innerOperands.push_back(memref);
            innerOperandTypes.push_back(memref.getType());
        });

        for (const auto& kvp : plan.entries) {
            Value memref = kvp.first;
            const pxa::CachePlan::Entry& entry = kvp.second;
            innerOperands.push_back(entry.cache);
            innerOperandTypes.push_back(entry.cache.getType());

            bool isOuter = entry.band == outer;
            EdslDMAStage stage = isOuter ? EdslDMAStage::OUTER : EdslDMAStage::MIDDLE;
            EdslDMADirection dir = getDirection(entry);
            if (dir == EdslDMADirection::IN) {
                inputs.push_back(memref);
            } else {
                outputs.push_back(memref);
                outputTypes.push_back(memref.getType());
            }
            auto idxs = isOuter ? outerIdxs.getArrayRef() : outerAndMiddleIdxs.getArrayRef();
            auto baseMap = getFlatAffineMap(context, entry.rap, idxs);
            Optional<int64_t> bufferId = getBufferIdentifier(memref);
            if (!bufferId) {
                memref.getDefiningOp()->emitOpError("Could not resolve buffer ID");
                return failure();
            }
            auto init = initializers.lookup(memref);
            if (init.value) {
                inits.push_back(init.value);
            } else {
                inits.push_back(builder.getUnitAttr());
            }
            auto dma = VPUIP_EdslDMADesc::get(AffineMapAttr::get(baseMap), EdslDMAStageAttr::get(context, stage),
                                              EdslDMADirectionAttr::get(context, dir), context);
            dmaDescs.push_back(dma);
            if (entry.copyFrom) {
                outputBufferIds.push_back(*bufferId);
            }
        }

        // What if a store is not cached? Do we have to DMA transfer the whole store
        // tensor, or direct access DDR? No real case yet.

        // Normalize and canonicalize the outer, middle, and inner loops.
        normalizeAffineParallel(outer);
        pxa::elideSingleIterationIndexes(outer);

        normalizeAffineParallel(middle);
        pxa::elideSingleIterationIndexes(middle);

        normalizeAffineParallel(inner);
        pxa::elideSingleIterationIndexes(inner);

        auto outerArgs = outer.getBody()->getArguments().vec();
        auto middleArgs = middle.getBody()->getArguments().vec();

        pxa::promoteIfEmptyIVs(middle);

        outer.walk([&](Operation* op) {
            if (failed(applyOpPatternsAndFold(op, canonPatterns))) {
                IVLOG(1, "Shave pipeline: canonicalization was failed.");
            }
        });

        // Construct a kernel FuncOp. Initially, the kernel is created inline so
        // that any operations defined above can be sunk into the kernel body.
        // Later, the kernel will be moved into a separate `vpux.module`.
        FunctionType kernelType = FunctionType::get(context, innerOperandTypes, inner.getResultTypes());
        std::string kernelName = llvm::formatv("{0}_kernel_{1}", funcName, kernelId).str();
        auto funcOp = builder.create<FuncOp>(outer.getLoc(), kernelName, kernelType);
        funcOp.addEntryBlock();
        KernelFuncHelper helper(funcOp, outerRanges.size(), middleRanges.size());

        // Clone the inner loop into the body of the kernel.
        BlockAndValueMapping mapping;
        mapping.map(outerArgs, helper.getOuterArgs());
        mapping.map(middleArgs, helper.getMiddleArgs());
        mapping.map(innerOperands, helper.getInnerArgs());
        OpBuilder kernelBuilder(funcOp.getBody());
        auto newInner = kernelBuilder.clone(*inner, mapping);
        kernelBuilder.create<ReturnOp>(outer.getLoc(), newInner->getResults());

        // Sink any operations that are defined above the scope of the inner loop.
        if (failed(sinkOperationsIntoKernel(funcOp))) {
            return failure();
        }

        FuncOp newFuncOp = moduleBuilder.cloneWithoutRegions(funcOp);
        newFuncOp.getBody().takeBody(funcOp.getBody());
        funcOp.erase();

        // Construct an EdslUPAOp.
        SmallVector<Attribute, 4> dmaAttrs(dmaDescs.begin(), dmaDescs.end());
        auto taskOp = builder.create<EdslUPAOp>(outer.getLoc(),
                                                /*inputs=*/inputs,
                                                /*outputs=*/outputs,
                                                /*kernel=*/builder.getSymbolRefAttr(newFuncOp),
                                                /*outers=*/builder.getI64ArrayAttr(outerRanges),
                                                /*middles=*/builder.getI64ArrayAttr(middleRanges),
                                                /*inits=*/builder.getArrayAttr(inits),
                                                /*transfers=*/builder.getArrayAttr(dmaAttrs));
        outer.replaceAllUsesWith(taskOp);

        for (auto item : llvm::zip(taskOp.getResults(), outputBufferIds)) {
            Value result;
            int64_t bufferId;
            std::tie(result, bufferId) = item;
            bufferIds.try_emplace(result, bufferId);
        }

        return success();
    }

    void assignBufferIdentifiers(FuncOp func) {
        unsigned nextId = 0;
        for (BlockArgument arg : func.getArguments()) {
            bufferIds.try_emplace(arg, nextId++);
        }
        for (AllocOp alloc : func.getOps<AllocOp>()) {
            bufferIds.try_emplace(alloc.getResult(), nextId++);
        }
    }

    Optional<int64_t> getBufferIdentifier(Value value) {
        auto it = bufferIds.find(value);
        if (it == bufferIds.end()) {
            return None;
        }
        return it->second;
    }
};

#endif

struct ShavePipelinePass : public ShavePipelineBase<ShavePipelinePass> {
    void runOnOperation() final {
#ifdef ENABLE_PLAIDML
        auto module = cast<ModuleOp>(*getOperation());
        SymbolTable symbolTable(module);
        auto vpuModule = ModuleOp::create(module.getLoc(), StringRef("kernels"));
        symbolTable.insert(vpuModule);
        for (auto func : module.getOps<FuncOp>()) {
            ShavePipelineImpl impl(func, vpuModule);
            if (failed(impl.runOnProgram(func.getName()))) {
                return signalPassFailure();
            }
        }
#else
        VPUX_THROW("ShavePipelinePass is only supported when ENABLE_PLAIDML=ON");
#endif
    }
};

std::unique_ptr<Pass> createShavePipelinePass() {
    return std::make_unique<ShavePipelinePass>();
}

}  // namespace edsl
}  // namespace vpux
