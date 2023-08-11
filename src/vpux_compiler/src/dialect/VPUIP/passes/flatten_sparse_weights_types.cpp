//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

// Create an explicit DeclareBuffer operand for the NCE weights operand to maintain the dense type
void setDenseWeightsOperandType(mlir::func::FuncOp func) {
    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.weights() == nullptr) {
            return;
        }

        auto weightsType = nceOp.weights().getType();
        if (VPUIP::getCompressionSchemeAttr(weightsType) == nullptr) {
            return;
        }

        auto declareOp = nceOp.weights().getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declareOp != nullptr, "Expected buffer declaration for weights, got {0}",
                          nceOp.weights().getDefiningOp());
        mlir::OpBuilder builder(declareOp);
        auto newDeclareOp = builder.clone(*declareOp.getOperation());
        declareOp.buffer().replaceUsesWithIf(newDeclareOp->getResult(0), [](mlir::OpOperand& operand) -> bool {
            return mlir::isa<VPUIP::NCEClusterTaskOp>(operand.getOwner());
        });
    });
}

// Enable the compression flag for the Sparsify transformation in order to flatten the constant's output type
void compressConstWeightsType(mlir::func::FuncOp func) {
    auto ctx = func.getContext();

    func.walk([&](Const::DeclareOp constOp) {
        const auto contentAttr = constOp.contentAttr();
        const auto transformations = contentAttr.getTransformations();
        const auto sparsifyTransformationIt =
                std::find_if(transformations.rbegin(), transformations.rend(), [](Const::TransformAttrInterface tr) {
                    return tr.isa<Const::SparsifyAttr>();
                });
        if (sparsifyTransformationIt == transformations.rend()) {
            return;
        }
        const auto sparsifyTransformation = sparsifyTransformationIt->dyn_cast<Const::SparsifyAttr>();

        const auto compressOutputType = mlir::BoolAttr::get(ctx, true);
        const auto newSparsifyTransformation =
                Const::SparsifyAttr::get(compressOutputType, sparsifyTransformation.getNumActualElements());

        SmallVector<Const::TransformAttrInterface> newTransformations;
        for (auto tr : transformations) {
            if (tr.isa<Const::SparsifyAttr>()) {
                newTransformations.push_back(newSparsifyTransformation);
                continue;
            }
            newTransformations.push_back(tr);
        }

        auto newOutputType = contentAttr.getBaseContent().getType().cast<vpux::NDTypeInterface>();
        for (auto tr : newTransformations) {
            newOutputType = tr.inferOutputType(newOutputType);
        }

        const auto newTransformationsRaw = to_small_vector(
                newTransformations | transformed([](vpux::Const::TransformAttrInterface attr) -> mlir::Attribute {
                    return attr;
                }));
        const auto newTransformationsAttr = mlir::ArrayAttr::get(ctx, newTransformationsRaw);

        auto newContentAttr =
                Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformationsAttr, newOutputType);
        constOp.contentAttr(newContentAttr);
    });
}

// Flatten the operand and result types of all operations, when the types contain a compression scheme
void flattenShapes(mlir::func::FuncOp func) {
    const auto eraseCompressionScheme = [](vpux::NDTypeInterface type) -> mlir::Type {
        if (auto memrefType = type.dyn_cast<mlir::MemRefType>()) {
            auto layout = memrefType.getLayout();
            if (auto memrefAttr = layout.dyn_cast<VPUIP::MemRefAttr>()) {
                layout = VPUIP::MemRefAttr::get(memrefAttr.order(), memrefAttr.strides(), memrefAttr.swizzlingScheme(),
                                                /*compressionScheme=*/nullptr, /*allocSize=*/nullptr,
                                                memrefAttr.getContext());
            }
            return mlir::MemRefType::get(memrefType.getShape(), memrefType.getElementType(), layout,
                                         memrefType.getMemorySpace());
        } else if (auto distType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
            return VPUIP::DistributedBufferType::get(
                    distType.getContext(), distType.getShape().raw(), distType.getElementType(), distType.getLayout(),
                    distType.getMemSpace(), distType.getDistribution(), /*compressionScheme=*/nullptr);
        } else if (auto bufferType = type.dyn_cast<VPUIP::BufferType>()) {
            return bufferType;
        }
        VPUX_THROW("Unsupported type {0}", type);
    };

    const auto flattenShape = [&](mlir::Value value) -> void {
        auto compressionScheme = VPUIP::getCompressionSchemeAttr(value.getType());
        if (compressionScheme == nullptr) {
            return;
        }
        const auto ndType = value.getType().cast<vpux::NDTypeInterface>();
        const auto totalByteSize = compressionScheme.getAllocSize(ndType.getElementType()).count();
        const Shape newShape({totalByteSize, 1, 1, 1});
        const auto flattenedType = ndType.changeShapeElemType(newShape, getUInt8Type(value.getContext()));
        const auto newType = eraseCompressionScheme(flattenedType);
        value.setType(newType);
    };

    const auto isNceWeightsOperand = [](mlir::Value value) {
        for (auto userOp : value.getUsers()) {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userOp);
            if (nceOp != nullptr && nceOp.weights() == value) {
                return true;
            }
        }
        return false;
    };

    func.walk([&](mlir::Operation* op) {
        for (auto operand : op->getOperands()) {
            if (isNceWeightsOperand(operand)) {
                continue;
            }
            flattenShape(operand);
        }
        for (auto result : op->getResults()) {
            if (isNceWeightsOperand(result)) {
                continue;
            }
            flattenShape(result);
        }
    });
}

//
// FlattenSparseWeightsTypes
//

class FlattenSparseWeightsTypes final : public VPUIP::FlattenSparseWeightsTypesBase<FlattenSparseWeightsTypes> {
public:
    explicit FlattenSparseWeightsTypes(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FlattenSparseWeightsTypes::safeRunOnFunc() {
    auto func = getOperation();

    setDenseWeightsOperandType(func);
    compressConstWeightsType(func);
    flattenShapes(func);
}

}  // namespace

//
// createFlattenSparseWeightsTypesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFlattenSparseWeightsTypesPass(Logger log) {
    return std::make_unique<FlattenSparseWeightsTypes>(log);
}
