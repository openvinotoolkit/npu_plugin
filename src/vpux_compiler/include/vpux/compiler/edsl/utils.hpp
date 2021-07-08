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

#pragma once

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace edsl {

static constexpr const char* kNoOptTag = "no_opt";
static constexpr const char* kTagAttribute = "tags";

struct Extent {
    int64_t min;
    int64_t max;
};

class PolynomialVisitor : public mlir::AffineExprVisitor<PolynomialVisitor> {
public:
    PolynomialVisitor(llvm::SmallVectorImpl<int64_t>& coeffs, llvm::SmallVectorImpl<int64_t>& dims)
            : coeffs(coeffs), dims(dims) {
    }

    void visitMulExpr(mlir::AffineBinaryOpExpr expr) {
        mlir::AffineExpr lhs = expr.getLHS();
        mlir::AffineExpr rhs = expr.getRHS();
        if (auto coeff = lhs.dyn_cast<mlir::AffineConstantExpr>()) {
            if (auto dim = rhs.dyn_cast<mlir::AffineDimExpr>()) {
                coeffs.emplace_back(coeff.getValue());
                dims.emplace_back(dim.getPosition());
            } else {
                throw std::runtime_error("Not a valid polynomial.");
            }
        } else if (auto coeff = rhs.dyn_cast<mlir::AffineConstantExpr>()) {
            if (auto dim = lhs.dyn_cast<mlir::AffineDimExpr>()) {
                coeffs.emplace_back(coeff.getValue());
                dims.emplace_back(dim.getPosition());
            } else {
                throw std::runtime_error("Not a valid polynomial.");
            }
        }
    }

    void visitAddExpr(mlir::AffineBinaryOpExpr expr) {
        visit(expr.getLHS());
        visit(expr.getRHS());
    }

    void visitDimExpr(mlir::AffineDimExpr expr) {
        coeffs.emplace_back(1);
        dims.emplace_back(expr.getPosition());
    }

    void visitConstantExpr(mlir::AffineConstantExpr expr) {
        coeffs.emplace_back(expr.getValue());
        dims.emplace_back(-1);
    }

private:
    llvm::SmallVectorImpl<int64_t>& coeffs;
    llvm::SmallVectorImpl<int64_t>& dims;
};

mlir::SmallVector<uint32_t, 4> padShapeTo4Dim(mlir::ArrayRef<int64_t> from);

template <typename Type>
mlir::SmallVector<Type, 4> getVectorFromArrayAttr(mlir::ArrayAttr attrs) {
    mlir::SmallVector<Type, 4> result;
    for (auto elem : attrs.getValue()) {
        result.emplace_back(elem.cast<mlir::IntegerAttr>().getInt());
    }
    return result;
}

MVCNN::DataType getSchemaDataType(mlir::Type type);

MVCNN::InitValue convertInitValue(mlir::Attribute attr);

// Input an expression and the value range of all variables. Return the extent
// of the expression
Extent computeExtent(mlir::AffineExpr expr, mlir::ArrayRef<Extent> vars);

// Check if all tags exist in op and they are all true
bool hasAllTags(mlir::Operation* op, llvm::ArrayRef<llvm::StringRef> tags);

// Check if tag exists in op
bool hasTag(mlir::Operation* op, llvm::StringRef tag);

// Set tags in op
void setTags(mlir::Operation* op, llvm::ArrayRef<llvm::StringRef> tags);

#ifdef ENABLE_PLAIDML

// Get the range of a loop index
int64_t loopIndexRange(mlir::BlockArgument idx);

// Get the step of a loop index
int64_t loopIndexStep(mlir::BlockArgument idx);

// If two memory accesses are same
// When ignoreMemRef is true, the caller assumes p's memref and q's memref are
// actually same (they could be differnt, but alias).
bool identicalMemoryAccess(mlir::Operation* p, mlir::Operation* q, bool ignoreMemRef = false);

#endif

}  // namespace edsl
}  // namespace vpux
