//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// PropagateCompressionScheme
//

class PropagateCompressionScheme final : public VPUIP::PropagateCompressionSchemeBase<PropagateCompressionScheme> {
public:
    explicit PropagateCompressionScheme(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    void reinferOutputType(mlir::Operation* op);
    void reinferInnerBlockTypes(VPUIP::NCEClusterTilingOp clusterTilingOp,
                                VPUIP::CompressionSchemeAttr compressionSchemeAttr);
    void propagateUpCompressionScheme(mlir::Value operand, VPUIP::CompressionSchemeAttr compressionSchemeAttr);
    void propagateDownCompressionScheme(mlir::Operation* op, VPUIP::CompressionSchemeAttr compressionSchemeAttr);
};

void PropagateCompressionScheme::reinferOutputType(mlir::Operation* op) {
    if (mlir::isa<mlir::InferTypeOpInterface>(op)) {
        vpux::inferReturnTypes(op, vpux::InferShapedTypeMode::ALL);
    } else if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
        for (auto p : op->getResults() | indexed) {
            auto resultIdx = p.index();
            auto result = p.value();
            auto outputOperand = VPUIP::getLayerViewSource(op, resultIdx);
            result.setType(outputOperand.getType());
        }
    }
}

// Reinfers the types inside the inner block of a cluster tiling operation so that the compact types
// contain the compression scheme of the outer operands
void PropagateCompressionScheme::reinferInnerBlockTypes(VPUIP::NCEClusterTilingOp clusterTilingOp,
                                                        VPUIP::CompressionSchemeAttr compressionSchemeAttr) {
    // Find the compact types for the new arguments and their locations
    SmallVector<mlir::Type> newArgTypes;
    SmallVector<mlir::Location> newArgLocations;
    auto& block = clusterTilingOp.body().front();
    const auto operandTypes = clusterTilingOp.getOperandTypes();
    const auto blockArgs = block.getArguments();
    for (auto p : zip(operandTypes, blockArgs)) {
        const auto operandType = std::get<0>(p);
        const auto arg = std::get<1>(p);
        newArgLocations.push_back(arg.getLoc());

        mlir::Type newArgType = operandType;
        if (auto distType = operandType.dyn_cast<VPUIP::DistributedBufferType>()) {
            newArgType = distType.getCompactType();
        } else if (auto sparseType = operandType.dyn_cast<VPUIP::SparseBufferType>()) {
            if (auto distDataType = sparseType.getData().dyn_cast<VPUIP::DistributedBufferType>()) {
                mlir::MemRefType dataType = distDataType.getCompactType();
                mlir::MemRefType smType = nullptr;
                if (sparseType.getSparsityMap() != nullptr &&
                    sparseType.getSparsityMap().isa<VPUIP::DistributedBufferType>()) {
                    smType = sparseType.getSparsityMap().cast<VPUIP::DistributedBufferType>().getCompactType();
                }
                mlir::MemRefType seType = nullptr;
                if (sparseType.getStorageElementTable() != nullptr &&
                    sparseType.getStorageElementTable().isa<VPUIP::DistributedBufferType>()) {
                    seType = sparseType.getStorageElementTable().cast<VPUIP::DistributedBufferType>().getCompactType();
                }
                newArgType = VPUIP::SparseBufferType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                                          sparseType.getCompressionScheme());
            }
        }
        newArgTypes.push_back(newArgType);
    }

    auto origArgCount = block.getArguments().size();

    // Add the new arguments and replace the uses of the original ones
    for (auto p : zip(newArgTypes, newArgLocations) | indexed) {
        auto type = std::get<0>(p.value());
        auto loc = std::get<1>(p.value());
        auto newArg = block.addArgument(type, loc);
        block.getArgument(p.index()).replaceAllUsesWith(newArg);
    }

    // Erase the original arguments
    while (origArgCount > 0) {
        block.eraseArgument(0);
        origArgCount--;
    }

    // Propagate the compression scheme inside the block of the cluster tiling operation
    auto firstOp = &block.front();
    propagateDownCompressionScheme(firstOp, compressionSchemeAttr);
}

// Propagates the compression scheme attribute upwards, until an operation without operands is reached (e.g. allocation)
void PropagateCompressionScheme::propagateUpCompressionScheme(mlir::Value operand,
                                                              VPUIP::CompressionSchemeAttr compressionSchemeAttr) {
    auto parentOp = operand.getDefiningOp();
    if (parentOp == nullptr || parentOp->getNumOperands() == 0) {
        auto newType = VPUIP::setCompressionSchemeAttr(operand.getType(), compressionSchemeAttr);
        operand.setType(newType);
        return;
    }

    if (mlir::isa<vpux::GroupedViewOpInterface>(parentOp)) {
        propagateUpCompressionScheme(parentOp->getOperand(0), compressionSchemeAttr);
    } else {
        for (auto operand : parentOp->getOperands()) {
            propagateUpCompressionScheme(operand, compressionSchemeAttr);
        }
    }

    reinferOutputType(parentOp);
}

// Propagates the compression scheme attribute to all user operations, until either an NCE operation is reached or the
// end of the model
void PropagateCompressionScheme::propagateDownCompressionScheme(mlir::Operation* op,
                                                                VPUIP::CompressionSchemeAttr compressionSchemeAttr) {
    if (mlir::isa<VPUIP::NCEClusterTaskOp, mlir::ReturnOp>(op)) {
        return;
    }

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
    if (clusterTilingOp != nullptr && mlir::isa<VPUIP::NCEClusterTaskOp>(clusterTilingOp.getInnerTaskOp())) {
        reinferInnerBlockTypes(clusterTilingOp, compressionSchemeAttr);
        return;
    }

    if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
        for (auto resultIdx : irange(op->getResults().size())) {
            auto outputOperand = VPUIP::getLayerViewSource(op, resultIdx);
            propagateUpCompressionScheme(outputOperand, compressionSchemeAttr);
        }
    }

    if (clusterTilingOp != nullptr) {
        reinferInnerBlockTypes(clusterTilingOp, compressionSchemeAttr);
    }

    reinferOutputType(op);

    for (auto userOp : op->getUsers()) {
        propagateDownCompressionScheme(userOp, compressionSchemeAttr);
    }
}

void PropagateCompressionScheme::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    func.walk([&](Const::DeclareOp constOp) {
        const auto contentAttr = constOp.contentAttr();
        const auto transformations = contentAttr.getTransformations();
        if (transformations.empty()) {
            return;
        }

        auto sparsifyTransformationIt =
                std::find_if(transformations.rbegin(), transformations.rend(), [](Const::TransformAttrInterface tr) {
                    return tr.isa<Const::SparsifyAttr>();
                });
        if (sparsifyTransformationIt == transformations.rend()) {
            return;
        }
        const auto sparsifyTransformation = sparsifyTransformationIt->dyn_cast<Const::SparsifyAttr>();

        const auto numElemsAttr = sparsifyTransformation.getNumActualElements();
        const auto axisAttr = getIntAttr(&ctx, Dims4D::Filter::OC.ind());
        const auto alignmentAttr = getIntAttr(&ctx, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
        auto compressionSchemeAttr = VPUIP::CompressionSchemeAttr::get(&ctx, axisAttr, numElemsAttr, alignmentAttr);

        const auto outputType = constOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = getMemRefType(
                outputType.getShape(), outputType.getElementType(), outputType.getDimsOrder(), outputType.getMemSpace(),
                outputType.getStrides(), vpux::getSwizzlingSchemeAttr(outputType), compressionSchemeAttr);

        constOp.output().setType(newOutputType);

        for (auto userOp : constOp.output().getUsers()) {
            auto groupOp = mlir::dyn_cast<VPUIP::GroupSparseBufferOp>(userOp);
            VPUX_THROW_UNLESS(groupOp != nullptr, "Expected weights user to be a VPUIP.GroupSparseBuffer op, got {0}",
                              userOp);
            VPUX_THROW_UNLESS(compressionSchemeAttr == groupOp.compression_schemeAttr(),
                              "Mismatch between the compression scheme of constant op '{0}' and grouping op '{1}'",
                              compressionSchemeAttr, groupOp.compression_schemeAttr());
            propagateDownCompressionScheme(userOp, compressionSchemeAttr);
        }
    });
}

}  // namespace

//
// createPropagateCompressionSchemePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createPropagateCompressionSchemePass(Logger log) {
    return std::make_unique<PropagateCompressionScheme>(log);
}
