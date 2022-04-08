//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/codec_factory.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {

//
// CompressWeightsBTCPass
//

class CompressWeightsBTCPass final : public VPUIP::CompressWeightsBTCBase<CompressWeightsBTCPass> {
public:
    explicit CompressWeightsBTCPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// NNDMAOpConverter
//

class NNDMAOpConverter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    NNDMAOpConverter(mlir::MLIRContext* ctx, const ICodec::CompressionAlgorithm& algo, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _codec(vpux::makeCodec(algo)), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    const std::unique_ptr<ICodec> _codec;
    VPU::ArchKind _arch;
    std::vector<uint8_t> compressDataFromDeclareOp(Const::DeclareOp constOp) const;
};

std::vector<uint8_t> NNDMAOpConverter::compressDataFromDeclareOp(Const::DeclareOp constOp) const {
    const auto content = constOp.content();
    const Byte totalInputSize = getTotalSize(constOp);
    std::vector<uint8_t> origData(checked_cast<size_t>(totalInputSize.count()));
    content.copyTo(makeMutableArrayRef(reinterpret_cast<char*>(origData.data()), origData.size()));

    return _codec->compress(origData);
}

mlir::LogicalResult NNDMAOpConverter::matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto loc = origOp->getLoc();
    auto input = origOp.input();
    auto output = origOp.output_buff();
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    auto inConstOp = input.getDefiningOp<Const::DeclareOp>();
    if (inConstOp == nullptr) {
        return mlir::failure();
    }

    auto outBufferOp = output.getDefiningOp<VPURT::DeclareBufferOp>();
    if (outBufferOp == nullptr) {
        return mlir::failure();
    }

    // dKMB weights compression only works with quantized data
    if (_arch != VPU::ArchKind::VPUX37XX) {
        const auto inContentAttr = inConstOp.contentAttr();
        const auto inContentType = inContentAttr.getType();
        if (!inContentType.getElementType().isa<mlir::quant::QuantizedType>()) {
            return mlir::failure();
        }
    }
    if (_arch == VPU::ArchKind::VPUX37XX || _arch == VPU::ArchKind::VPUX30XX) {
        if (outputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
            _log.trace("CompressedDMA only support CONST2CMX on {0} platform", _arch);
            return mlir::failure();
        }
    }

    _log.trace("Check if can change to compressed DMA, operation - '{0}'", loc);

    const auto originInShape = inputType.getShape().raw();
    const auto originOutShape = outputType.getShape().raw();

    const auto strideInReqs = StrideReqs::compact(originInShape.size());
    const auto strideOutReqs = StrideReqs::compact(originOutShape.size());

    if (!strideInReqs.checkStrides(input) || !strideOutReqs.checkStrides(output)) {
        _log.nest().trace("Strides check failed");
        return mlir::failure();
    }

    if (outputType.isa<VPUIP::DistributedBufferType>()) {
        const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
        const auto distributionAttr = distributedType.getDistribution();
        const auto distributionMode = distributionAttr.mode().getValue();
        if (distributionMode != VPU::DistributionMode::DUPLICATED) {
            _log.nest().trace("Only DUPLICATE Distributed mode supported, mode - '{0}'",
                              VPU::stringifyDistributionMode(distributionMode));
            return mlir::failure();
        }
    }

    const Byte totalInputSize = getTotalSize(origOp.input());
    constexpr Byte MIN_INPUT_SIZE = 4_KB;
    if (totalInputSize < MIN_INPUT_SIZE) {
        _log.nest().trace("Size smaller than minimal '{0}' < '{1}'", totalInputSize.count(), MIN_INPUT_SIZE.count());
        return mlir::failure();
    }

    _log.trace("Compress constant '{0}', type - '{1}'", inConstOp->getLoc(), inputType);

    const auto compressedData = compressDataFromDeclareOp(inConstOp);
    if (compressedData.empty()) {
        _log.nest().trace("Compression failed");
        return mlir::failure();
    }

    const auto ctx = rewriter.getContext();
    const auto u8Type = getUInt8Type(ctx);
    const Shape flatDstShape{checked_cast<int64_t>(totalInputSize.count()), 1, 1, 1};
    auto newDstType = outputType.changeShapeElemType(flatDstShape, u8Type);
    newDstType = newDstType.changeDimsOrder(DimsOrder::NCHW);

    rewriter.setInsertionPointAfter(outBufferOp);
    auto newDstBufferOp = rewriter.create<VPURT::DeclareBufferOp>(
            outBufferOp->getLoc(), newDstType, outBufferOp.sectionAttr(), outBufferOp.sectionIndexAttr(),
            outBufferOp.byteOffsetAttr(), outBufferOp.swizzlingKeyAttr());

    const Shape flatSrcShape{checked_cast<int64_t>(compressedData.size()), 1, 1, 1};
    const auto newSrcStorageType = mlir::RankedTensorType::get(flatSrcShape.raw(), u8Type);
    const auto newSrcContentAttr = mlir::DenseElementsAttr::get(newSrcStorageType, makeArrayRef(compressedData));
    const auto newSrcType = getMemRefType(flatSrcShape, u8Type, DimsOrder::NCHW, inputType.getMemSpace());

    rewriter.setInsertionPointAfter(inConstOp);
    auto newSrcConstOp = rewriter.create<Const::DeclareOp>(inConstOp->getLoc(), newSrcType,
                                                           Const::ContentAttr::get(newSrcContentAttr));

    rewriter.setInsertionPoint(origOp);
    rewriter.create<VPUIP::CompressedDMAOp>(loc, newSrcConstOp.output(), newDstBufferOp.buffer(), origOp.portAttr(),
                                            origOp.is_out_of_orderAttr(), origOp.is_criticalAttr());
    rewriter.replaceOp(origOp, {outBufferOp.buffer()});

    const auto uncompressed = totalInputSize.count();
    const auto compressed = compressedData.size();
    _log.trace("Compressed weights for {0}: {1} / {2} ({3})", loc, compressed, uncompressed,
               (double)compressed / uncompressed);

    return mlir::success();
}

void CompressWeightsBTCPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    // NOTE: the pass is enabled only for VPUX37XX for now
    if (arch != VPU::ArchKind::VPUX37XX) {
        return;
    }

    const auto algo = (arch == VPU::ArchKind::VPUX37XX) ? ICodec::CompressionAlgorithm::BITCOMPACTOR_CODEC
                                                        : ICodec::CompressionAlgorithm::HUFFMAN_CODEC;

    _log.trace("VPUIP CompressWeightsBTCPass");
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<NNDMAOpConverter>(&ctx, algo, arch, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCompressWeightsBTCPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCompressWeightsBTCPass(Logger log) {
    return std::make_unique<CompressWeightsBTCPass>(log);
}
