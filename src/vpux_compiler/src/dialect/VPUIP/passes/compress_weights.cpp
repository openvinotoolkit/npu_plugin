//
// Copyright Intel Corporation.
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
    auto declareConstOp = origOp.input().getDefiningOp<Const::DeclareOp>();
    if (!declareConstOp) {
        return mlir::failure();
    }
    auto declareTensorOp = origOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    if (!declareTensorOp) {
        return mlir::failure();
    }
    const auto attr = declareConstOp.contentAttr();
    const auto type = attr.getType();
    // TODO find out whether other data types can be compressed.
    if (!type.getElementType().isa<mlir::quant::QuantizedType>()) {
        return mlir::failure();
    }
    constexpr Byte MIN_INPUT_SIZE = 4_KB;
    const Byte totalInputSize = getTotalSize(origOp.input());
    if (totalInputSize < MIN_INPUT_SIZE) {
        return mlir::failure();
    }
    const size_t totalByteSize = totalInputSize.count();
    const auto content = attr.fold();

    std::vector<uint8_t> dataVec(alignVal(totalByteSize, sizeof(uint64_t)), 0);

    const auto buf = makeMutableArrayRef(reinterpret_cast<char*>(dataVec.data()), totalByteSize);
    content.copyTo(buf);

    const auto compressedData = _codec->compress(dataVec);
    if (compressedData.empty()) {
        return mlir::failure();
    }

    auto origDstTensor = declareTensorOp;
    const auto location = origDstTensor.locale();
    const auto localeIndex = origDstTensor.localeIndex();
    const auto dataIndex = origDstTensor.dataIndex();
    const auto dstMemSpace = origDstTensor.getType().cast<mlir::MemRefType>().getMemorySpace();
    // TODO find out whether the destination shape also has to be flat.
    const auto elemTypeU8 = getUInt8Type(rewriter.getContext());
    const Shape flatDstTensorShape{checked_cast<int64_t>(dataVec.size()), 1, 1, 1};
    auto dstTensorType = getMemRefType(flatDstTensorShape, elemTypeU8, DimsOrder::NCHW, dstMemSpace);
    auto dstTensor = rewriter.create<VPURT::DeclareBufferOp>(origOp->getLoc(), dstTensorType, location,
                                                             parseIntArrayAttr<int64_t>(localeIndex), dataIndex);
    const Shape flatSrcTensorShape{checked_cast<int64_t>(compressedData.size()), 1, 1, 1};
    const auto dataStorageType = mlir::RankedTensorType::get(flatSrcTensorShape.raw(), elemTypeU8);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, mlir::ArrayRef<uint8_t>(compressedData));

    const auto srcMemSpace = origOp.input().getType().cast<mlir::MemRefType>().getMemorySpace();
    const auto dataType = getMemRefType(flatSrcTensorShape, elemTypeU8, DimsOrder::NCHW, srcMemSpace);

    auto srcTensor = rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataType, Const::ContentAttr::get(dataAttr));
    // Compressed tensor must be flat because of the check in the inference runtime.
    const auto srcTensorShape = getShape(srcTensor).raw();
    const auto nonTrivialDims =
            std::count_if(srcTensorShape.begin(), srcTensorShape.end(), [](const int64_t& dim) -> bool {
                return dim > 1;
            });
    VPUX_THROW_WHEN(nonTrivialDims > 1, "NNDMAOpConverter::matchAndRewrite: source tensor is not flat");
    // Introducing CompressedDMAOp which drops IERT_SameShape interface, since it is not applicable for compression.
    rewriter.create<VPUIP::CompressedDMAOp>(origOp->getLoc(), srcTensor, dstTensor);
    _log.trace("Compressing weights for {0}", origOp->getLoc());
    rewriter.replaceOp(origOp, {declareTensorOp});

    return mlir::success();
}

void CompressWeightsPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    // FIXME enable compression via attribute
    const auto algo = (arch == VPU::ArchKind::MTL) ? ICodec::CompressionAlgorithm::BITCOMPACTOR_CODEC
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
