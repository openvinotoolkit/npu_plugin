//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
namespace {

//
// FuseConstants
//

class FuseConstants final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    FuseConstants(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::RankedTensorType populateFusedConstantBuffer(vpux::ConstantFusing::ConstantVector& constantVector,
                                                       std::vector<uint8_t>& fusedValuesBuf,
                                                       mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::Value createAllocOp(Const::DeclareOp declOp, VPURT::AllocDistributed allocDistributed,
                          mlir::PatternRewriter& rewriter) {
    if (allocDistributed) {
        auto origType = allocDistributed.getType().cast<VPUIP::DistributedBufferType>();
        auto newType = vpux::ConstantFusing::getDistributedBufferType(origType, declOp, rewriter);
        auto distributedBufferType = newType.cast<VPUIP::DistributedBufferType>();
        return rewriter.create<VPURT::AllocDistributed>(declOp.getLoc(), distributedBufferType, nullptr, nullptr)
                .getBuffer();

    } else {
        const auto type = declOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        vpux::IndexedSymbolAttr memKindAttr =
                IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
        auto newType = type.changeMemSpace(memKindAttr);
        auto memrefType = newType.cast<mlir::MemRefType>();
        return rewriter.create<mlir::memref::AllocOp>(declOp.getLoc(), memrefType).getMemref();
    }
}

VPUIP::CopyOp createFusedCopyOp(mlir::Value allocDefiningOp, Const::DeclareOp declOp, mlir::PatternRewriter& rewriter) {
    VPUIP::CopyOp fusedCopyOp = nullptr;
    if (auto allocOp = allocDefiningOp.getDefiningOp<VPURT::AllocDistributed>()) {
        SmallVector<mlir::Value> inputsOutputOperands = {declOp.getResult(), allocOp.getBuffer()};
        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            fusedCopyOp = builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        rewriter.create<VPUIP::NCEClusterTilingOp>(appendLoc(declOp.getLoc(), "_fused_tile"), allocDefiningOp.getType(),
                                                   inputsOutputOperands, bodyBuilder);
    } else if (auto allocOp = allocDefiningOp.getDefiningOp<mlir::memref::AllocOp>()) {
        fusedCopyOp = rewriter.create<VPUIP::CopyOp>(declOp->getLoc(), declOp.getOutput(), allocOp.getMemref());
    } else {
        VPUX_THROW("Unrecognized allocDefiningOp encountered");
    }
    return fusedCopyOp;
}

mlir::RankedTensorType FuseConstants::populateFusedConstantBuffer(vpux::ConstantFusing::ConstantVector& constantVector,
                                                                  std::vector<uint8_t>& fusedValuesBuf,
                                                                  mlir::PatternRewriter& rewriter) const {
    int64_t totalTensorsize = 0;
    for (auto& constant : constantVector) {
        if (constant.second != nullptr) {
            totalTensorsize += vpux::getTotalSize(constant.second->getOpResult(0)).count();
        }
    }

    fusedValuesBuf.reserve(totalTensorsize);
    unsigned jointF16constantsSize = 0;

    for (auto& pair : constantVector) {
        // In case of some layers like MaxPool the weights won't be present so skip over to the next
        // constant for fusion
        if (pair.second != nullptr) {
            auto content = pair.second.getContent();
            auto contentType = pair.second.getType().cast<vpux::NDTypeInterface>();
            auto elemType = contentType.getElementType();
            auto origElemType = elemType;

            if (VPUIP::getCompressionSchemeAttr(contentType) != nullptr) {
                elemType = getUInt8Type(elemType.getContext());
            }
            // Currently assuming that Quant types are 8 bit integers
            // To be updated for future FP8, INT4, FP4 support
            if (auto quantElemType = elemType.dyn_cast<mlir::quant::QuantizedType>()) {
                if (quantElemType.isSigned()) {
                    auto values = content.getValues<int8_t>();
                    for (size_t idx = 0; idx < values.size(); ++idx) {
                        auto int8Val = values[idx];
                        fusedValuesBuf.push_back(*reinterpret_cast<uint8_t*>(&int8Val));
                    }
                } else {
                    auto values = content.getValues<uint8_t>();
                    for (size_t idx = 0; idx < values.size(); ++idx) {
                        fusedValuesBuf.push_back(values[idx]);
                    }
                }
            } else if (elemType.isUnsignedInteger(8)) {
                auto values = content.getValues<uint8_t>();
                for (size_t idx = 0; idx < values.size(); ++idx) {
                    fusedValuesBuf.push_back(values[idx]);
                }
                if (origElemType.isF16()) {
                    jointF16constantsSize += values.size();
                }
            } else if (elemType.isF16()) {
                auto weightsValues = content.getValues<float16>();
                for (size_t idx = 0; idx < weightsValues.size(); ++idx) {
                    vpux::ConstantFusing::convertToU8<float16>(weightsValues[idx], fusedValuesBuf);
                }
                jointF16constantsSize += weightsValues.size() * 2;
            } else if (elemType.isSignedInteger(32)) {
                auto weightTableValues = content.getValues<int32_t>();
                for (size_t idx = 0; idx < weightTableValues.size(); ++idx) {
                    vpux::ConstantFusing::convertToU8<int32_t>(weightTableValues[idx], fusedValuesBuf);
                }
            } else if (elemType.isInteger(1)) {
                const auto packedNumElems = contentType.getNumElements() / CHAR_BIT;
                const auto packedElemType = getUInt8Type(contentType.getContext());
                const auto packedContentType =
                        contentType.changeShapeElemType(Shape({1, 1, 1, packedNumElems}), packedElemType);
                auto packedContent = Const::Content::fromRawBuffer(packedContentType, content.getRawStorageBuf(),
                                                                   packedElemType, content.isSplat());
                auto weightsSMValues = packedContent.getValues<uint8_t>();
                for (size_t idx = 0; idx < weightsSMValues.size(); ++idx) {
                    fusedValuesBuf.push_back(weightsSMValues[idx]);
                }
            } else {
                VPUX_THROW("Unsupported data type for constant {0}", pair.second.getLoc());
            }
        }
    }

    mlir::RankedTensorType fusedTensorType = nullptr;
    if (jointF16constantsSize * 2 > totalTensorsize) {  // use F16 type if the fused constant
                                                        // is to be dominated by this type
        auto fusedConstantElementType = mlir::FloatType::getF16(rewriter.getContext());
        SmallVector<int64_t> fusedConstShape({1, 1, 1, totalTensorsize / 2});

        fusedTensorType = mlir::RankedTensorType::get(fusedConstShape, fusedConstantElementType);
    } else {
        auto fusedConstantElementType = getUInt8Type(rewriter.getContext());
        SmallVector<int64_t> fusedConstShape({1, 1, 1, totalTensorsize});

        fusedTensorType = mlir::RankedTensorType::get(fusedConstShape, fusedConstantElementType);
    }

    return fusedTensorType;
}

void replaceConstantsWithFusedConstant(vpux::ConstantFusing::ConstantVector& constantVector,
                                       vpux::ConstantFusing::TilingOpVector& tilingVector,
                                       mlir::PatternRewriter& rewriter, VPUIP::CopyOp newCopyOp) {
    VPUIP::NCEClusterTilingOp newTilingOp = newCopyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    auto opElementType = newCopyOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto opElementSizeBytes = opElementType.getIntOrFloatBitWidth() / CHAR_BIT;

    // 5.  Replace constants constant with sequence fused_constant -> subview -> view
    int64_t offset = 0;
    vpux::Byte size(0);
    for (size_t i = 0; i < constantVector.size(); ++i) {
        auto constant = constantVector[i].second;
        if (constant == nullptr) {
            continue;
        }
        auto constTilingOp = tilingVector[i].second;
        size = vpux::getTotalSize(constant->getOpResult(0)) / opElementSizeBytes;
        SmallVector<int64_t> subtensor({1, 1, 1, size.count()});
        auto offsets = SmallVector<int64_t>{0, 0, 0, offset};
        if (constTilingOp != nullptr) {
            auto subViewOp =
                    rewriter.create<VPUIP::SubViewOp>(constant.getLoc(), newTilingOp->getResult(0), offsets, subtensor);
            rewriter.replaceOpWithNewOp<VPUIP::ViewOp>(constTilingOp, constTilingOp.getOutputBuffs()[0].getType(),
                                                       subViewOp.getResult());
        } else {
            auto copyOp = constantVector[i].first;
            auto subViewOp =
                    rewriter.create<VPUIP::SubViewOp>(constant.getLoc(), newCopyOp.getOutput(), offsets, subtensor);
            rewriter.replaceOpWithNewOp<VPUIP::ViewOp>(copyOp, copyOp.getOutputBuff().getType(), subViewOp.getResult());
        }
        offset += size.count();
    }
}

// For a given layer type we need to determine the constant fusing order given the presence (or not) of weights,
// weights sparsity map, weight table and activation window. In certain cases it might not be possible to fuse
// the constants e.g for case when layer weights are not constants and are in graphfile or if the declare or copyop
// couldn't be found in such case matchFailed is returned with the error message
// E#45170 - Update the logic to make constant selection generic
mlir::LogicalResult getInputsInFusingOrder(VPUIP::NCEClusterTaskOp& nceOp,
                                           vpux::ConstantFusing::ConstantVector& constantVector,
                                           vpux::ConstantFusing::TilingOpVector& tilingVector,
                                           mlir::PatternRewriter& rewriter) {
    VPUIP::CopyOp copyOp = nullptr;
    Const::DeclareOp declareOp = nullptr;

    VPURT::AllocDistributed allocDistributed = nullptr;
    VPUIP::NCEClusterTilingOp tilingOp = nullptr;

    auto resetTemporaries = [&]() {
        copyOp = nullptr;
        declareOp = nullptr;
        allocDistributed = nullptr;
        tilingOp = nullptr;
    };

    auto hasOneUse = [](VPUIP::CopyOp copyOp) {
        auto tilingCopyOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        return tilingCopyOp == nullptr ? copyOp->hasOneUse() : tilingCopyOp->hasOneUse();
    };

    vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp, nceOp.getWeightTable(), copyOp, declareOp,
                                                       allocDistributed, tilingOp);

    if (copyOp != nullptr && declareOp != nullptr) {
        constantVector[0] = {copyOp, declareOp};
        tilingVector[0] = {allocDistributed, tilingOp};
    } else {
        return matchFailed(rewriter, nceOp, "Couldn't find weight table");
    }

    if (nceOp.getWeights() != nullptr) {
        resetTemporaries();
        vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp, nceOp.getWeights(), copyOp, declareOp,
                                                           allocDistributed, tilingOp);
        if (copyOp == nullptr) {
            return matchFailed(rewriter, nceOp, "Weights Copy Op missing");
        }
        // TODO Handling shared weights with weight table. Ticket E#61458
        if (!hasOneUse(copyOp)) {
            return matchFailed(rewriter, nceOp, "Weights Copy Op has more than one use");
        }

        if (declareOp != nullptr) {
            constantVector[1] = {copyOp, declareOp};
            tilingVector[1] = {allocDistributed, tilingOp};
        } else {
            // Special condition when weights come in from a different source
            // e.g. Activation tensor
            return matchFailed(rewriter, nceOp, "Non constant layer weights");
        }
    }

    if (nceOp.getWeightsSparsityMap() != nullptr) {
        resetTemporaries();
        vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp, nceOp.getWeightsSparsityMap(), copyOp, declareOp,
                                                           allocDistributed, tilingOp);
        if (copyOp == nullptr) {
            return matchFailed(rewriter, nceOp, "Weights sparsity map Copy Op missing");
        }

        if (!hasOneUse(copyOp)) {
            return matchFailed(rewriter, nceOp, "Weights sparsity copy op has more than one use");
        }

        if (declareOp != nullptr) {
            constantVector[2] = {copyOp, declareOp};
            tilingVector[2] = {allocDistributed, tilingOp};
        } else {
            return matchFailed(rewriter, nceOp, "The layer weights sparsity map is not constant");
        }
    }

    if (nceOp.getActivationWindow() != nullptr) {
        resetTemporaries();
        vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp, nceOp.getActivationWindow(), copyOp, declareOp,
                                                           allocDistributed, tilingOp);
        if (declareOp != nullptr && copyOp != nullptr) {
            if (!hasOneUse(copyOp)) {
                return matchFailed(rewriter, nceOp, "Weights activation window copy op has more than one use");
            }

            constantVector[3] = {copyOp, declareOp};
            tilingVector[3] = {allocDistributed, tilingOp};
        }
    }
    return mlir::success();
}

mlir::LogicalResult FuseConstants::matchAndRewrite(VPUIP::NCEClusterTaskOp nceOp,
                                                   mlir::PatternRewriter& rewriter) const {
    if (nceOp->hasAttr(vpux::ConstantFusing::constantsFused)) {
        return mlir::failure();
    }

    if (auto parentTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        rewriter.setInsertionPoint(parentTilingOp);
    }

    if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE || nceOp.getTaskType() == VPUIP::NCETaskType::AVEPOOL) {
        return mlir::failure();
    }

    // 1. Find constant inputs
    vpux::ConstantFusing::ConstantVector constantVector(vpux::ConstantFusing::numberOfConstantsToFuse,
                                                        {nullptr, nullptr});
    vpux::ConstantFusing::TilingOpVector tilingVector(vpux::ConstantFusing::numberOfConstantsToFuse,
                                                      {nullptr, nullptr});
    if (getInputsInFusingOrder(nceOp, constantVector, tilingVector, rewriter).failed()) {
        constantVector.clear();
        tilingVector.clear();
        return mlir::failure();
    }

    // 2. Create fused constant of u8 type with size of weights + weights sparsity map + weights table + activation
    // window Fill it with the original binary data
    const auto newLoc = appendLoc(nceOp.getLoc(), "_fused_constant");
    std::vector<uint8_t> fusedValuesBuf;
    auto tensorType = populateFusedConstantBuffer(constantVector, fusedValuesBuf, rewriter);
    VPUX_THROW_UNLESS(tensorType != nullptr, "Couldn't fuse constant tensor type");

    // 3. Build new constant memref
    auto fusedTensorTypeMemref = vpux::convertToMemRef(tensorType);
    mlir::ElementsAttr value;
    if (tensorType.getElementType().isF16()) {
        auto f16Type = mlir::FloatType::getF16(rewriter.getContext());
        unsigned f16TypeSizeBytes = f16Type.getWidth() / CHAR_BIT;
        value = mlir::DenseElementsAttr::get(tensorType,
                                             ArrayRef<float16>(reinterpret_cast<const float16*>(fusedValuesBuf.data()),
                                                               fusedValuesBuf.size() / f16TypeSizeBytes));
    } else {
        value = mlir::DenseElementsAttr::get(tensorType, ArrayRef(fusedValuesBuf));
    }
    auto fusedConstant =
            rewriter.create<Const::DeclareOp>(newLoc, fusedTensorTypeMemref, Const::ContentAttr::get(value));

    // 4. build new AllocOp
    auto allocOp = createAllocOp(fusedConstant, tilingVector[0].first, rewriter);

    // 5. create CopyOp, copy constant to allocated buffer
    auto copyOp = createFusedCopyOp(allocOp, fusedConstant, rewriter);

    // 6.  Replace constants with sequence fused_constant -> subview -> viewOp
    replaceConstantsWithFusedConstant(constantVector, tilingVector, rewriter, copyOp);

    // 7. Set constantsFused attribute so we can check the fusion status (fused or unfused) of the current layer
    // in patch_weight_table pass in VPUIP Dialect
    nceOp->setAttr(vpux::ConstantFusing::constantsFused, mlir::BoolAttr::get(nceOp.getContext(), true));
    return mlir::success();
}

//
// FuseConstantsPass
//

class FuseConstantsPass final : public VPUIP::FuseConstantsBase<FuseConstantsPass> {
public:
    explicit FuseConstantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseConstantsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FuseConstants>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseConstantsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFuseConstantsPass(Logger log) {
    return std::make_unique<FuseConstantsPass>(log);
}
