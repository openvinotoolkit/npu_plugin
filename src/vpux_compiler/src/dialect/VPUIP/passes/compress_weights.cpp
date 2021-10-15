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

#include "bitCompactor.h"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {

class BitCompactorCodec {
public:
    BitCompactorCodec();
    std::vector<uint8_t> compress(std::vector<uint8_t>& data) const;

private:
    // Classified mutable because BitCompactor::btcmpctr_cmprs_bound is non-constant.
    // FIXME Make the method constant in bitcompactor repository.
    mutable BitCompactor _bitCompactor;
};

BitCompactorCodec::BitCompactorCodec(): _bitCompactor() {
    _bitCompactor.mBitCompactorConfig->blockSize = 64;
    _bitCompactor.mBitCompactorConfig->superBlockSize = 4096;
    _bitCompactor.mBitCompactorConfig->minFixedBitLn = 3;
    _bitCompactor.mBitCompactorConfig->cmprs = 1;
    _bitCompactor.mBitCompactorConfig->bypass_en = false;
    _bitCompactor.mBitCompactorConfig->dual_encode_en = true;
    _bitCompactor.mBitCompactorConfig->proc_bin_en = false;
    _bitCompactor.mBitCompactorConfig->proc_btmap_en = false;
    _bitCompactor.mBitCompactorConfig->mixedBlkSize = false;
    _bitCompactor.mBitCompactorConfig->align = 1;
    _bitCompactor.mBitCompactorConfig->ratio = false;
    _bitCompactor.mBitCompactorConfig->verbosity = 0;  // set between 0-5,
                                                       // 0 shows basic info,
                                                       // 3 shows Metadata and some other useful stuff,
                                                       // 5 shows all available info
}

std::vector<uint8_t> BitCompactorCodec::compress(std::vector<uint8_t>& data) const {
    VPUX_THROW_UNLESS(!data.empty(), "BitCompactorCodec::compress: Empty input data vector");

    BitCompactor::btcmpctr_compress_wrap_args_t btcArgs;

    btcArgs.bypass_en = _bitCompactor.mBitCompactorConfig->bypass_en;
    btcArgs.dual_encode_en = _bitCompactor.mBitCompactorConfig->dual_encode_en;
    btcArgs.proc_bin_en = _bitCompactor.mBitCompactorConfig->proc_bin_en;
    btcArgs.proc_btmap_en = _bitCompactor.mBitCompactorConfig->proc_btmap_en;
    btcArgs.align = _bitCompactor.mBitCompactorConfig->align;
    btcArgs.verbosity = _bitCompactor.mBitCompactorConfig->verbosity;
    btcArgs.SblkSize = _bitCompactor.mBitCompactorConfig->blockSize;
    btcArgs.LblkSize = _bitCompactor.mBitCompactorConfig->superBlockSize;
    btcArgs.mixedBlkSize = _bitCompactor.mBitCompactorConfig->mixedBlkSize;
    btcArgs.minFixedBitLn = _bitCompactor.mBitCompactorConfig->minFixedBitLn;

    auto uncompressedDataSize = checked_cast<int32_t>(data.size());
    auto compressedBufferSizeBound = _bitCompactor.btcmpctr_cmprs_bound(uncompressedDataSize);

    std::vector<uint8_t> compressedDataBuffer(compressedBufferSizeBound, 0);
    auto compressedSize = _bitCompactor.CompressArray(data.data(), uncompressedDataSize, compressedDataBuffer.data(),
                                                      compressedBufferSizeBound, &btcArgs);
    // Trim trailing bytes.
    compressedDataBuffer.resize(compressedSize);

    // sometimes even if the tensor is > 4KB it might not be compressable
    VPUX_THROW_UNLESS(uncompressedDataSize >= compressedSize, "BitCompactorCodec::compress: Compression failed.");

    return compressedDataBuffer;
}

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
    NNDMAOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _codec() {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    const BitCompactorCodec _codec;
};

mlir::LogicalResult NNDMAOpConverter::matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    auto declareConstOp = origOp.input().getDefiningOp<Const::DeclareOp>();
    if (!declareConstOp) {
        return mlir::failure();
    }
    auto declareTensorOp = origOp.output_buff().getDefiningOp<VPUIP::DeclareTensorOp>();
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

    auto compressedData = _codec.compress(dataVec);

    auto origDstTensor = declareTensorOp;
    const auto location = origDstTensor.locale();
    const auto localeIndex = origDstTensor.localeIndex();
    const auto dataIndex = origDstTensor.dataIndex();
    const auto dstMemSpace = origDstTensor.getType().cast<mlir::MemRefType>().getMemorySpace();
    // TODO find out whether the destination shape also has to be flat.
    const auto elemTypeU8 = getUInt8Type(rewriter.getContext());
    const SmallVector<int64_t> flatDstTensorShape{checked_cast<int64_t>(dataVec.size()), 1, 1, 1};
    auto dstTensorType = changeMemSpace(mlir::MemRefType::get(flatDstTensorShape, elemTypeU8), dstMemSpace);
    auto dstTensor = rewriter.create<VPUIP::DeclareTensorOp>(origOp->getLoc(), dstTensorType, location,
                                                             parseIntArrayAttr<int64_t>(localeIndex), dataIndex);
    const SmallVector<int64_t> flatSrcTensorShape{checked_cast<int64_t>(compressedData.size()), 1, 1, 1};
    const auto dataStorageType = mlir::RankedTensorType::get(flatSrcTensorShape, elemTypeU8);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, mlir::ArrayRef<uint8_t>(compressedData));

    const auto srcMemSpace = origOp.input().getType().cast<mlir::MemRefType>().getMemorySpace();
    const auto dataType = changeMemSpace(mlir::MemRefType::get(flatSrcTensorShape, elemTypeU8), srcMemSpace);

    auto srcTensor = rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataType, Const::ContentAttr::get(dataAttr));
    // Compressed tensor must be flat because of the check in the inference runtime.
    const auto srcTensorShape = getShape(srcTensor).raw();
    const auto nonTrivialDims =
            std::count_if(srcTensorShape.begin(), srcTensorShape.end(), [](const int64_t& dim) -> bool {
                return dim > 1;
            });
    VPUX_THROW_WHEN(nonTrivialDims > 1, "NNDMAOpConverter::matchAndRewrite: source tensor is not flat");
    // Introducing CompressedDMAOp which drops IERT_SameShape interface, since it is not applicable for compression.
    rewriter.create<VPUIP::CompressedDMAOp>(origOp->getLoc(), srcTensor, dstTensor, origOp.waitBarriers(),
                                            origOp.updateBarriers());
    _log.trace("Compressing weights for {0}", origOp->getLoc());
    rewriter.replaceOp(origOp, {declareTensorOp});

    return mlir::success();
}

void CompressWeightsPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPUIP::getArch(module);
    if (arch != VPUIP::ArchKind::MTL) {
        _log.trace("Weights compression is supported only for MTL");
        return;
    }
    // FIXME enable compression via attribute

    _log.trace("VPUIP CompressWeightsPass");
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<NNDMAOpConverter>(&ctx, _log);

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
