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
// CompressWeightsPass
//

class CompressWeightsPass final : public VPUIP::CompressWeightsBase<CompressWeightsPass> {
public:
    explicit CompressWeightsPass(Logger log) {
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
    NNDMAOpConverter(mlir::MLIRContext* ctx, const ICodec::CompressionAlgorithm& algo, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _codec(vpux::makeCodec(algo)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    const std::unique_ptr<ICodec> _codec;
};

mlir::LogicalResult NNDMAOpConverter::matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    auto inConstOp = origOp.input().getDefiningOp<Const::DeclareOp>();
    if (inConstOp == nullptr) {
        return mlir::failure();
    }

    auto outBufferOp = origOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    if (outBufferOp == nullptr) {
        return mlir::failure();
    }

    const auto inContentAttr = inConstOp.contentAttr();
    const auto inContentType = inContentAttr.getType();

    // TODO find out whether other data types can be compressed.
    if (!inContentType.getElementType().isa<mlir::quant::QuantizedType>()) {
        return mlir::failure();
    }

    constexpr Byte MIN_INPUT_SIZE = 4_KB;
    const Byte totalInputSize = getTotalSize(origOp.input());
    if (totalInputSize < MIN_INPUT_SIZE) {
        return mlir::failure();
    }

    const auto inContent = inContentAttr.fold();
    std::vector<uint8_t> origData(checked_cast<size_t>(totalInputSize.count()));
    inContent.copyTo(makeMutableArrayRef(reinterpret_cast<char*>(origData.data()), origData.size()));

    const auto compressedData = _codec->compress(origData);
    if (compressedData.empty()) {
        return mlir::failure();
    }

    const auto elemTypeU8 = getUInt8Type(rewriter.getContext());

    // TODO find out whether the destination shape also has to be flat.
    const Shape flatDstShape{checked_cast<int64_t>(origData.size()), 1, 1, 1};
    const auto outBuffType = outBufferOp.getType().cast<vpux::NDTypeInterface>();
    const auto newDstType = getMemRefType(flatDstShape, elemTypeU8, DimsOrder::NCHW, outBuffType.getMemSpace());

    auto newDstBufferOp =
            rewriter.create<VPURT::DeclareBufferOp>(origOp->getLoc(), newDstType, outBufferOp.sectionAttr(),
                                                    outBufferOp.sectionIndexAttr(), outBufferOp.byteOffsetAttr());

    const Shape flatSrcShape{checked_cast<int64_t>(compressedData.size()), 1, 1, 1};
    const auto newSrcStorageType = mlir::RankedTensorType::get(flatSrcShape.raw(), elemTypeU8);
    const auto newSrcContentAttr = mlir::DenseElementsAttr::get(newSrcStorageType, makeArrayRef(compressedData));
    const auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto newSrcType = getMemRefType(flatSrcShape, elemTypeU8, DimsOrder::NCHW, inType.getMemSpace());

    auto newSrcConstOp =
            rewriter.create<Const::DeclareOp>(origOp->getLoc(), newSrcType, Const::ContentAttr::get(newSrcContentAttr));

    _log.trace("Compressing weights for {0}", origOp->getLoc());
    rewriter.create<VPUIP::CompressedDMAOp>(origOp->getLoc(), newSrcConstOp.output(), newDstBufferOp.buffer(),
                                            origOp.portAttr(), origOp.is_out_of_orderAttr(), origOp.is_criticalAttr());
    rewriter.replaceOp(origOp, {outBufferOp.buffer()});

    return mlir::success();
}

void CompressWeightsPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    // FIXME enable compression via attribute
    const auto algo = (arch == VPU::ArchKind::VPUX37XX) ? ICodec::CompressionAlgorithm::BITCOMPACTOR_CODEC
                                                        : ICodec::CompressionAlgorithm::HUFFMAN_CODEC;

    _log.trace("VPUIP CompressWeightsPass");
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<NNDMAOpConverter>(&ctx, algo, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCompressWeightsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCompressWeightsPass(Logger log) {
    return std::make_unique<CompressWeightsPass>(log);
}
