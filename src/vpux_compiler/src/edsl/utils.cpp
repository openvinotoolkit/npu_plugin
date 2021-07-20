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

#include "vpux/compiler/edsl/utils.hpp"

#include <mlir/Support/DebugStringHelper.h>

#ifdef ENABLE_PLAIDML
#include "pmlc/dialect/pxa/ir/ops.h"
#endif

using namespace mlir;

#ifdef ENABLE_PLAIDML
namespace pxa = pmlc::dialect::pxa;
#endif

namespace vpux {
namespace edsl {

class ExtentComputer : public AffineExprVisitor<ExtentComputer, Extent> {
public:
    explicit ExtentComputer(ArrayRef<Extent> extents): vars(extents) {
    }

    Extent visitMulExpr(AffineBinaryOpExpr expr) {
        Extent lhs = computeExtent(expr.getLHS(), vars);
        Extent rhs = computeExtent(expr.getRHS(), vars);
        std::array<int64_t, 3> values = {lhs.min * rhs.max, lhs.max * rhs.min, lhs.max * rhs.max};
        Extent result = {lhs.min * rhs.min, lhs.min * rhs.min};
        for (int64_t value : values) {
            if (value < result.min) {
                result.min = value;
            }
            if (value > result.max) {
                result.max = value;
            }
        }
        return result;
    }

    Extent visitAddExpr(AffineBinaryOpExpr expr) {
        Extent lhs = computeExtent(expr.getLHS(), vars);
        Extent rhs = computeExtent(expr.getRHS(), vars);
        return {lhs.min + rhs.min, lhs.max + rhs.max};
    }

    Extent visitDimExpr(AffineDimExpr expr) {
        unsigned pos = expr.getPosition();
        if (pos >= vars.size()) {
            throw std::runtime_error("Position exceeds the size of vars");
        }
        return vars[pos];
    }

    Extent visitConstantExpr(AffineConstantExpr expr) {
        int64_t value = expr.getValue();
        return {value, value};
    }

    Extent visitSymbolExpr(AffineSymbolExpr) {
        throw std::runtime_error("Unexpected affine expresssion: SymbolExpr.");
    }

    Extent visitCeilDivExpr(AffineBinaryOpExpr) {
        throw std::runtime_error("Unexpected affine expresssion: CeilDivExpr.");
    }

    Extent visitFloorDivExpr(AffineBinaryOpExpr) {
        throw std::runtime_error("Unexpected affine expresssion: FloorDivExpr.");
    }

    Extent visitModExpr(AffineBinaryOpExpr) {
        throw std::runtime_error("Unexpected affine expresssion: ModExpr.");
    }

private:
    ArrayRef<Extent> vars;
};

Extent computeExtent(AffineExpr expr, ArrayRef<Extent> vars) {
    ExtentComputer ec(vars);
    return ec.visit(expr);
}

SmallVector<uint32_t, 4> padShapeTo4Dim(ArrayRef<int64_t> from) {
    SmallVector<uint32_t, 4> into;
    if (from.size() <= 4) {
        for (unsigned i = 0; i < 4 - from.size(); ++i) {
            into.emplace_back(1);
        }
    }
    for (auto dim : from) {
        into.emplace_back(static_cast<uint32_t>(dim));
    }
    return into;
}

MVCNN::DataType getSchemaDataType(Type type) {
    if (type.isF16())
        return MVCNN::DataType_FLOAT16;
    if (type.isF32())
        return MVCNN::DataType_FLOAT32;
    if (type.isF64())
        return MVCNN::DataType_FLOAT64;
    if (type.isSignlessInteger()) {
        switch (type.getIntOrFloatBitWidth()) {
        case 8:
            return MVCNN::DataType_UINT8;
        case 16:
            return MVCNN::DataType_UINT16;
        case 32:
            return MVCNN::DataType_UINT32;
        case 64:
            return MVCNN::DataType_UINT64;
        }
    }
    VPUX_THROW("Unsupported DType: {0}", debugString(type));
}

MVCNN::InitValue convertInitValue(Attribute attr) {
    if (attr) {
        if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            return MVCNN::InitValue{/*needInit=*/true, /*isInt=*/true,
                                    /*intValue=*/intAttr.getInt(),
                                    /*floatValue=*/0.0};
        }
        if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
            return MVCNN::InitValue{/*needInit=*/true, /*isInt=*/false,
                                    /*intValue=*/0,
                                    /*floatValue=*/floatAttr.getValueAsDouble()};
        }
    }
    return MVCNN::InitValue{};
}

// Check if all tags exist
bool hasAllTags(Operation* op, ArrayRef<StringRef> tags) {
    if (tags.empty()) {
        return true;
    }
    DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
    if (!opTagsAttr) {
        return false;
    }
    for (StringRef tag : tags) {
        if (!opTagsAttr.get(tag)) {
            return false;
        }
    }
    return true;
}

bool hasTag(Operation* op, StringRef tag) {
    DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
    if (!opTagsAttr) {
        return false;
    }
    return opTagsAttr.get(tag) != nullptr;
}

// Set tags in op
void setTags(Operation* op, ArrayRef<StringRef> tags) {
    if (tags.empty()) {
        return;
    }
    OpBuilder builder(op);
    DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
    SmallVector<NamedAttribute, 4> newTags;
    if (opTagsAttr) {
        newTags.append(opTagsAttr.begin(), opTagsAttr.end());
    }
    for (StringRef tag : tags) {
        if (!opTagsAttr || !opTagsAttr.get(tag)) {
            newTags.emplace_back(builder.getNamedAttr(tag, builder.getUnitAttr()));
        }
    }
    op->setAttr(kTagAttribute, builder.getDictionaryAttr(newTags));
}

#ifdef ENABLE_PLAIDML

int64_t loopIndexRange(BlockArgument idx) {
    if (auto loop = dyn_cast<AffineParallelOp>(idx.getOwner()->getParentOp())) {
        auto idxs = loop.getBody()->getArguments();
        auto ranges = loop.getConstantRanges().getValue();
        for (unsigned i = 0; i < idxs.size(); ++i) {
            if (idxs[i] == idx) {
                return ranges[i];
            }
        }
    }
    return -1;
}

int64_t loopIndexStep(BlockArgument idx) {
    if (auto loop = dyn_cast<AffineParallelOp>(idx.getOwner()->getParentOp())) {
        auto idxs = loop.getBody()->getArguments();
        auto steps = loop.getSteps();
        for (unsigned i = 0; i < idxs.size(); ++i) {
            if (idxs[i] == idx) {
                return steps[i];
            }
        }
    }
    return -1;
}

bool equalLoopIdxs(ValueRange idxs0, ValueRange idxs1) {
    if (idxs0.size() != idxs1.size()) {
        return false;
    }
    for (unsigned i = 0; i < idxs0.size(); ++i) {
        if (loopIndexRange(idxs0[i].cast<BlockArgument>()) != loopIndexRange(idxs1[i].cast<BlockArgument>()) ||
            loopIndexStep(idxs0[i].cast<BlockArgument>()) != loopIndexStep(idxs1[i].cast<BlockArgument>())) {
            return false;
        }
    }
    return true;
}

bool identicalMemoryAccess(Operation* p, Operation* q, bool ignoreMemRef) {
    if (auto load0 = dyn_cast<pxa::PxaLoadOp>(p)) {
        if (auto load1 = dyn_cast<pxa::PxaLoadOp>(q)) {
            if ((ignoreMemRef || load0.getMemRef() == load1.getMemRef()) &&
                load0.getAffineMap() == load1.getAffineMap() && equalLoopIdxs(load0.idxs(), load1.idxs())) {
                return true;
            }
        }
    } else if (auto reduce0 = dyn_cast<pxa::PxaReduceOp>(p)) {
        if (auto reduce1 = dyn_cast<pxa::PxaReduceOp>(q)) {
            if ((ignoreMemRef || reduce0.getMemRef() == reduce1.getMemRef()) &&
                reduce0.getAffineMap() == reduce1.getAffineMap() && equalLoopIdxs(reduce0.idxs(), reduce1.idxs())) {
                return true;
            }
        }
    }
    return false;
}

#endif

}  // namespace edsl
}  // namespace vpux
