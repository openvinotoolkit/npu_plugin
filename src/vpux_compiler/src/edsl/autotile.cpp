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

#include "vpux/compiler/edsl/autotile.hpp"

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/IntegerSet.h>

#ifdef ENABLE_PLAIDML
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"
#endif

#include "vpux/compiler/edsl/passes.hpp"
#include "vpux/compiler/edsl/utils.hpp"

using namespace mlir;          // NOLINT
using namespace mlir::memref;  // NOLINT

namespace vpux {
namespace edsl {

#ifdef ENABLE_PLAIDML

namespace pxa = pmlc::dialect::pxa;

struct Condition {
    // Idxs' positions in the loop
    SmallVector<uint32_t, 8> idxs;
    // Idxs' coefficients
    SmallVector<int64_t, 8> coeffs;
    int64_t constant;
};

// Look for the negative items in a polynomial
class NegativeItemVisitor : public AffineExprVisitor<NegativeItemVisitor> {
public:
    void visitAddExpr(AffineBinaryOpExpr expr) {
        visit(expr.getLHS());
        visit(expr.getRHS());
    }

    void visitMulExpr(AffineBinaryOpExpr expr) {
        auto lhs = expr.getLHS();
        auto rhs = expr.getRHS();
        if (auto coeff = lhs.dyn_cast<AffineConstantExpr>()) {
            if (coeff.getValue() < 0) {
                if (auto dim = rhs.dyn_cast<AffineDimExpr>()) {
                    negativeDims.insert(dim.getPosition());
                }
            }
        } else if (auto coeff = rhs.dyn_cast<AffineConstantExpr>()) {
            if (coeff.getValue() < 0) {
                if (auto dim = lhs.dyn_cast<AffineDimExpr>()) {
                    negativeDims.insert(dim.getPosition());
                }
            }
        }
    }

    void init() {
        negativeDims.clear();
    }
    llvm::SmallSet<uint32_t, 16>& getNegativeDims() {
        return negativeDims;
    }

private:
    llvm::SmallSet<uint32_t, 16> negativeDims;
};

struct AutoTileInfo {
    SmallVector<Operation*, 4> tensorOps;
    SmallVector<int64_t, 8> ranges;  // The ranges are divided by steps
    SmallVector<bool, 8> innerIdxOnly;
    SmallVector<bool, 8> outerIdxOnly;
    SmallVector<Condition, 4> conditions;
    uint32_t vectorFactor;
    uint64_t insideAllocatedSize;
    AutoTileParams params;

    static Optional<AutoTileInfo> create(AffineParallelOp op, const AutoTileParams& params) {
        auto ranges = op.getConstantRanges();
        if (!ranges) {
            return None;
        }

        AutoTileInfo info;
        info.params = params;

        auto steps = op.getSteps();
        for (unsigned i = 0; i < ranges->size(); ++i) {
            info.ranges.emplace_back(llvm::divideCeil((*ranges)[i], steps[i]));
        }

        info.collectTensors(op);
        info.estimateVectorFactor(op);

        return info;
    }

    void insertTensorOp(Operation* newOp) {
        // Do not insert the duplicated memory access
        for (auto op : tensorOps) {
            if (identicalMemoryAccess(op, newOp)) {
                return;
            }
        }
        tensorOps.emplace_back(newOp);
    }

    // Collect the indices in the negative items
    void collectNegativeIdxs(AffineMap map, ValueRange idxs, llvm::SmallPtrSet<void*, 16>& negativeIdxs) {
        NegativeItemVisitor visitor;
        for (auto expr : map.getResults()) {
            visitor.init();
            visitor.visit(expr);
            llvm::SmallSet<uint32_t, 16>& negDims = visitor.getNegativeDims();
            for (auto dim : negDims) {
                negativeIdxs.insert(idxs[dim].getAsOpaquePointer());
            }
        }
    }

    void collectTensors(AffineParallelOp op) {
        // The indices appear in the output memory accesses
        llvm::SmallPtrSet<void*, 16> outputIdxs;
        // The indices appear in the negative items in the memory accesses
        llvm::SmallPtrSet<void*, 16> negativeIdxs;
        llvm::SmallPtrSet<Operation*, 8> insideMemUses;
        tensorOps.clear();
        innerIdxOnly.clear();
        outerIdxOnly.clear();
        insideAllocatedSize = 0;

        // Collect the operations that use the buffer defined inside op
        op.walk([&](AllocOp alloc) {
            for (auto& use : pxa::getIndirectUses(alloc.getResult())) {
                insideMemUses.insert(use.getOwner());
            }
            insideAllocatedSize += pmlc::util::getByteSize(alloc.getType());
        });

        op.walk([&](Operation* subOp) {
            if (auto load = dyn_cast<pxa::PxaLoadOp>(subOp)) {
                if (insideMemUses.count(load.getOperation()) == 0) {
                    insertTensorOp(load);
                    collectNegativeIdxs(load.getAffineMap(), load.idxs(), negativeIdxs);
                }
            } else if (auto reduce = dyn_cast<pxa::PxaReduceOp>(subOp)) {
                if (insideMemUses.count(reduce.getOperation()) == 0) {
                    insertTensorOp(reduce);
                    for (auto idx : reduce.idxs()) {
                        outputIdxs.insert(idx.getAsOpaquePointer());
                    }
                    collectNegativeIdxs(reduce.getAffineMap(), reduce.idxs(), negativeIdxs);
                }
            }
        });

        // Set innerIdxOnly and outerIdxOnly for each idx
        DenseMap<void*, uint32_t> idxPosition;
        for (unsigned i = 0; i < op.getBody()->getNumArguments(); ++i) {
            auto idx = op.getBody()->getArgument(i);
            innerIdxOnly.emplace_back(params.outputIndicesOnly ? outputIdxs.count(idx.getAsOpaquePointer()) == 0
                                                               : false);
            bool outputOnly = false;
            if (params.accIndicesOnly && outputIdxs.count(idx.getAsOpaquePointer()) > 0) {
                outputOnly = true;
            } else if (params.noNegativeIndex && negativeIdxs.count(idx.getAsOpaquePointer()) > 0) {
                outputOnly = true;
            }
            outerIdxOnly.emplace_back(outputOnly);
            idxPosition.try_emplace(idx.getAsOpaquePointer(), i);
        }

        // Collect conditions for If operation
        for (auto ifOp : op.getOps<AffineIfOp>()) {
            auto constraints = ifOp.getIntegerSet();
            for (auto expr : constraints.getConstraints()) {
                SmallVector<int64_t, 8> coeffs;
                SmallVector<int64_t, 8> dims;
                PolynomialVisitor visitor(coeffs, dims);
                visitor.visit(expr);
                Condition cond;
                for (unsigned i = 0; i < dims.size(); ++i) {
                    if (dims[i] < 0) {
                        cond.constant = coeffs[i];
                        continue;
                    }
                    Value idx = ifOp.getOperand(dims[i]);
                    uint32_t pos = idxPosition[idx.getAsOpaquePointer()];
                    cond.idxs.emplace_back(pos);
                    cond.coeffs.emplace_back(coeffs[i]);
                }
                conditions.emplace_back(cond);
            }
        }
    }

    void estimateVectorFactor(AffineParallelOp op) {
        unsigned width = 1;
        for (auto& subOp : llvm::make_early_inc_range(op.region().front())) {
            unsigned opWidth;
            if (auto load = dyn_cast<pxa::PxaLoadOp>(subOp)) {
                opWidth = Byte(getElemTypeSize(load.getResult().getType())).count();
            } else if (auto reduce = dyn_cast<pxa::PxaReduceOp>(subOp)) {
                opWidth = Byte(getElemTypeSize(reduce.val().getType())).count();
            } else if (auto cast = dyn_cast<IndexCastOp>(subOp)) {
                auto result = cast.getResult();
                opWidth = result.getType().isIndex() ? 1 : Byte(getElemTypeSize(cast.getResult().getType())).count();
            } else {
                continue;
            }
            width = std::max(opWidth, width);
        }
        vectorFactor = params.vectorWidth / width;
    }
};

struct SingleShaveCostModel {
    SingleShaveCostModel(AffineParallelOp op, const AutoTileInfo& info): loopOp(op), info(info) {
    }

    SmallVector<int64_t, 4> computeTileShape(ArrayRef<int64_t> tile, MemRefType tensorType, AffineMap affineMap,
                                             ValueRange idxs) const {
        auto loopIdxs = loopOp.getBody()->getArguments();
        SmallVector<Extent, 8> ranges;
        for (Value idx : idxs) {
            unsigned k = 0;
            while (k < loopIdxs.size() && loopIdxs[k] != idx) {
                ++k;
            }
            if (k < loopIdxs.size()) {
                // Loop argument
                ranges.push_back(Extent{0, tile[k] - 1});
            } else if (auto argIdx = idx.dyn_cast<BlockArgument>()) {
                // Block argument
                ranges.push_back(Extent{0, loopIndexRange(argIdx) - 1});
            } else {
                // Arbitrary index, could be the result of index_cast
                // Use the max of int32_t to avoid int64_t overflow during extent computation
                ranges.push_back(Extent{0, std::numeric_limits<int32_t>::max()});
            }
        }
        auto exprs = affineMap.getResults();
        auto shape = tensorType.getShape();
        SmallVector<int64_t, 4> results;
        for (unsigned i = 0; i < exprs.size(); ++i) {
            Extent extent = computeExtent(exprs[i], ranges);
            if (extent.max >= shape[i]) {
                extent.max = shape[i] - 1;
            }
            // Don't allow negative index
            results.push_back((extent.min >= 0) ? (extent.max - extent.min + 1) : (extent.max + 1));
        }
        return results;
    }

    uint64_t getUsedBuffer(ArrayRef<int64_t> shape, Value memRef) const {
        MemRefType memType = memRef.getType().dyn_cast<MemRefType>();
        unsigned eltSize = memType.getElementType().getIntOrFloatBitWidth();
        uint32_t size = llvm::divideCeil(eltSize, 8);
        for (auto dim : shape) {
            size *= dim;
        }
        return size;
    }

    double getIOCost(ArrayRef<int64_t> shape, Value memRef) const {
        MemRefType memType = memRef.getType().dyn_cast<MemRefType>();
        ArrayRef<int64_t> memShape = memType.getShape();
        if (shape.size() != memShape.size()) {
            throw std::runtime_error("Autotile VPUX: tiled dim is different from the "
                                     "original tensor dim.");
        }
        double cache_elems =
                static_cast<double>(info.params.cacheWidth) * 8 / memType.getElementType().getIntOrFloatBitWidth();
        double cache_lines = 1.0;
        int64_t max_val = 0;
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            int64_t gap = std::abs(stride) - max_val;
            cache_lines *= static_cast<double>(shape[i]);
            double prob_shared = 0.0;  // Assume it's never shared
            if (cache_elems != 0.0 && gap < cache_elems) {
                prob_shared = 1.0 - (gap / cache_elems);
            }
            cache_lines -= prob_shared * static_cast<double>(shape[i] - 1);
            max_val += std::abs(stride) * (shape[i] - 1);
            stride *= memShape[i];
        }
        return cache_lines;
    }

    // Return the total computation and the vectorized total computation
    double getComputeCost(ArrayRef<int64_t> tile, bool vectorized = false) const {
        double count = 1.0;
        for (unsigned i = 0; i < tile.size() - 1; ++i) {
            count *= tile[i];
        }
        // Let the vectorization factor <= the last dim
        uint32_t factor = info.vectorFactor;
        while (factor > tile.back()) {
            factor >>= 1;
        }
        return vectorized ? (count * ((tile.back() - 1) / factor + 1)) : (count * tile.back());
    }

    // For each fixed outer loop idxs, test if info.conditions are satisfied for
    // all inner iterations, or unsatisfied for all inner iterations
    bool fullySatisfiedOrNot(ArrayRef<int64_t> tile) const {
        // Compute the extents for all conditions using tile (i.e., inner idxs)
        SmallVector<Extent, 8> extents;
        for (auto& condition : info.conditions) {
            Extent extent = {0, 0};
            for (unsigned i = 0; i < condition.idxs.size(); ++i) {
                if (condition.coeffs[i] > 0) {
                    extent.max += condition.coeffs[i] * (tile[condition.idxs[i]] - 1);
                } else {
                    extent.min += condition.coeffs[i] * (tile[condition.idxs[i]] - 1);
                }
            }
            extents.emplace_back(extent);
        }
        // Traverse all possible outer loop idxs
        SmallVector<int64_t, 8> outerIdxs(tile.size(), -1);
        int pos = 0;
        while (pos >= 0) {
            if (pos >= static_cast<int>(tile.size())) {
                // We have the outer idxs. Then test all conditions
                for (unsigned k = 0; k < info.conditions.size(); ++k) {
                    auto& condition = info.conditions[k];
                    int64_t result = condition.constant;
                    for (unsigned i = 0; i < condition.idxs.size(); ++i) {
                        result += condition.coeffs[i] * outerIdxs[i];
                    }
                    if (result + extents[k].min < 0 && result + extents[k].max >= 0) {
                        // Not all iterations are satisfied or not satisfied
                        return false;
                    }
                }
                --pos;
                continue;
            }
            if (outerIdxs[pos] == -1) {
                outerIdxs[pos] = 0;
            } else {
                outerIdxs[pos] += tile[pos];
            }
            if (outerIdxs[pos] >= info.ranges[pos]) {
                outerIdxs[pos] = -1;
                --pos;
            } else {
                ++pos;
            }
        }
        return true;
    }

    double operator()(ArrayRef<int64_t> tile, double bestCost) const {
        int64_t outerCount = 1;
        for (unsigned i = 0; i < tile.size(); ++i) {
            if (info.innerIdxOnly[i] && tile[i] < info.ranges[i]) {
                // This index can be only inner index so that tile[i] must be
                // outerRange[i]
                return std::numeric_limits<double>::infinity();
            }
            if (info.outerIdxOnly[i] && tile[i] > 1) {
                // This index can be only outer index so that tile[i] must be 1
                return std::numeric_limits<double>::infinity();
            }
            outerCount *= ((info.ranges[i] - 1) / tile[i] + 1);
        }
        if (outerCount > info.params.maxCount || outerCount < info.params.minCount) {
            return std::numeric_limits<double>::infinity();
        }

        uint64_t bufferSize = info.insideAllocatedSize;
        double ioCost = 0;
        for (unsigned i = 0; i < info.tensorOps.size(); ++i) {
            Operation* op = info.tensorOps[i];
            AffineMap affineMap;
            ValueRange idxs;
            Value memRef;
            if (auto load = dyn_cast<pxa::PxaLoadOp>(op)) {
                affineMap = load.getAffineMap();
                idxs = load.getMapOperands();
                memRef = load.getMemRef();
            } else if (auto reduce = dyn_cast<pxa::PxaReduceOp>(op)) {
                affineMap = reduce.getAffineMap();
                idxs = reduce.getMapOperands();
                memRef = reduce.getMemRef();
            }
            auto shape = computeTileShape(tile, memRef.getType().cast<MemRefType>(), affineMap, idxs);
            bufferSize += getUsedBuffer(shape, memRef);
            ioCost += getIOCost(shape, memRef);
        }
        if (bufferSize > info.params.totalBuffer) {
            return std::numeric_limits<double>::infinity();
        }
        double efficiency = 1;
        if (bufferSize < info.params.minInnerBuffer) {
            efficiency *= (static_cast<double>(info.params.minInnerBuffer) / static_cast<double>(bufferSize));
        }
        double computeCost = getComputeCost(tile);
        double newCost = efficiency * ioCost / computeCost;
        if (newCost > bestCost) {
            // Can't be the best plan. Skip the following test.
            return newCost;
        }
        if (info.conditions.size() == 0 || fullySatisfiedOrNot(tile)) {
            // For the fixed outer loop idxs, the conditions are satisfied for all
            // inner iterations, or they are unsatisfied for all inner iterations
            return newCost;
        }
        return std::numeric_limits<double>::infinity();
    }

    mutable AffineParallelOp loopOp;
    const AutoTileInfo& info;
};

struct EvenTilingGenerator {
    std::vector<int64_t> operator()(int64_t range) const {
        std::vector<int64_t> out;
        for (int64_t r = 1; r <= range; r++) {
            if (range % r != 0) {
                continue;
            }
            out.push_back(r);
        }
        return out;
    }
};

SmallVector<int64_t, 8> computeBestTile(AffineParallelOp op, const AutoTileParams& params) {
    auto info = AutoTileInfo::create(op, params);
    if (!info) {
        return {};
    }
    SingleShaveCostModel costModel(op, *info);
    EvenTilingGenerator generator;
    return pxa::findBestTileSize(generator, costModel, info->ranges);
}

#endif

struct AutoTileVPUX : public AutoTileVPUXBase<AutoTileVPUX> {
    AutoTileVPUX() {
    }

    AutoTileVPUX(const AutoTileParams& params): params(params) {
    }

    void runOnFunction() final {
#ifdef ENABLE_PLAIDML
        if (!params.hasValue()) {
            params = AutoTileParams{
                    /*totalBuffer=*/totalBuffer.getValue(),
                    /*minCount=*/minCount.getValue(),
                    /*maxCount=*/maxCount.getValue(),
                    /*minInnferBuffer=*/minInnerBuffer.getValue(),
                    /*cacheWidth=*/cacheWidth.getValue(),
                    /*vectorWidth=*/vectorWidth.getValue(),
                    /*processingTags=*/processingTags.getValue(),
                    /*outerTags=*/outerTags.getValue(),
                    /*innerTags=*/innerTags.getValue(),
                    /*outputIndicesOnly=*/outputIndicesOnly.getValue(),
                    /*accIndicesOnly=*/accIndicesOnly.getValue(),
                    /*noNegativeIndex=*/noNegativeIndex.getValue(),
            };
        }

        processingTagList.clear();
        llvm::SplitString(params->processingTags, processingTagList, ",");

        outerTagList.clear();
        llvm::SplitString(params->outerTags, outerTagList, ",");

        innerTagList.clear();
        llvm::SplitString(params->innerTags, innerTagList, ",");

        auto func = getFunction();
        if (hasTag(func, kNoOptTag)) {
            return;
        }

        func.walk([&](AffineParallelOp op) {
            if (!hasAllTags(op, processingTagList)) {
                return;
            }

            auto tileSize = computeBestTile(op, params.getValue());
            if (tileSize.empty()) {
                return;
            }

            pxa::performTiling(op, tileSize);

            // Set tags for outer loop and inner loop
            setTags(op, outerTagList);
            for (auto inner : op.getOps<AffineParallelOp>()) {
                setTags(inner, innerTagList);
            }
        });
#else
        VPUX_THROW("AutoTileVPUX is only supported when ENABLE_PLAIDML=ON");
#endif
    }

    Optional<AutoTileParams> params;
    SmallVector<StringRef, 4> processingTagList;
    SmallVector<StringRef, 4> outerTagList;
    SmallVector<StringRef, 4> innerTagList;
};

std::unique_ptr<Pass> createAutoTileVPUXPass(const AutoTileParams& params) {
    return std::make_unique<AutoTileVPUX>(params);
}

std::unique_ptr<Pass> createAutoTileVPUXPass() {
    return std::make_unique<AutoTileVPUX>();
}

}  // namespace edsl
}  // namespace vpux
