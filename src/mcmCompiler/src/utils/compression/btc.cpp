#include "include/mcm/utils/compression/btc.hpp"
#include <vector>

namespace mv
{
BTC::BTC(uint32_t align, uint32_t bitmapPreprocEnable, bool pStatsOnly, bool bypassMode, uint32_t verbosity)
{
    codec_.reset(new BitCompactor());

    setDefaultConfig();

    updateConfig(align,
                bitmapPreprocEnable,
                pStatsOnly,
                bypassMode,
                verbosity
                );
}

void BTC::setDefaultConfig()
{
    codec_->mBitCompactorConfig->blockSize = 64;
    codec_->mBitCompactorConfig->superBlockSize = 4096;
    codec_->mBitCompactorConfig->minFixedBitLn = 3;
    codec_->mBitCompactorConfig->cmprs = 1;
    codec_->mBitCompactorConfig->bypass_en = false;
    codec_->mBitCompactorConfig->dual_encode_en = true;
    codec_->mBitCompactorConfig->proc_bin_en = false;
    codec_->mBitCompactorConfig->proc_btmap_en = false;
    codec_->mBitCompactorConfig->mixedBlkSize = false;
    codec_->mBitCompactorConfig->align = 1;
    codec_->mBitCompactorConfig->ratio = false;
    codec_->mBitCompactorConfig->verbosity = 0;     // set between 0-5,
                                            // 0 shows basic info,
                                            // 3 shows Metadata and some other useful stuff,
                                            // 5 shows all available info

}

void BTC::updateConfig(uint32_t align, uint32_t bitmapPreprocEnable, bool pStatsOnly, bool bypassMode, uint32_t verbosity)
{
    codec_->mBitCompactorConfig->bypass_en = bypassMode;
    codec_->mBitCompactorConfig->verbosity = verbosity;
    codec_->mBitCompactorConfig->bypass_en = bypassMode;
    codec_->mBitCompactorConfig->proc_btmap_en = bitmapPreprocEnable;
    codec_->mBitCompactorConfig->align = align;
}

std::pair<std::vector<int64_t>, uint32_t> BTC::compress(std::vector<int64_t>& data, mv::Tensor& t)
{
    if(data.empty()) {
        throw mv::ArgumentError("hde", "hdeCompress", "0", "Empty data vector");
    }

    BitCompactor::btcmpctr_compress_wrap_args_t btcArgs;

    btcArgs.bypass_en      = codec_->mBitCompactorConfig->bypass_en;
    btcArgs.dual_encode_en = codec_->mBitCompactorConfig->dual_encode_en;
    btcArgs.proc_bin_en    = codec_->mBitCompactorConfig->proc_bin_en;
    btcArgs.proc_btmap_en  = codec_->mBitCompactorConfig->proc_btmap_en;
    btcArgs.align          = codec_->mBitCompactorConfig->align;
    btcArgs.verbosity      = codec_->mBitCompactorConfig->verbosity;
    btcArgs.SblkSize       = codec_->mBitCompactorConfig->blockSize;
    btcArgs.LblkSize       = codec_->mBitCompactorConfig->superBlockSize;
    btcArgs.mixedBlkSize   = codec_->mBitCompactorConfig->mixedBlkSize;
    btcArgs.minFixedBitLn  = codec_->mBitCompactorConfig->minFixedBitLn;

    std::pair<std::vector<int64_t>, uint32_t> toReturn;
    std::vector<uint8_t> uncompressedData(data.begin(),data.end());
    uint32_t uncompressedDataSize = uncompressedData.size();
    //auto compressedBufferSize = uncompressedDataSize + BTC_MAX_COMPRESS_FACTOR * (std::ceil(uncompressedDataSize / 4096.0) + 1);

    auto compressedBufferSizeBound = codec_->btcmpctr_cmprs_bound(uncompressedDataSize);

    std::vector<uint8_t> compressedDataBuffer (compressedBufferSizeBound, 0);
    uint32_t compressedSize = codec_->CompressArray(&uncompressedData[0], uncompressedDataSize, &compressedDataBuffer[0], compressedBufferSizeBound, &btcArgs);
    std::vector<uint8_t>::iterator endDataIterator = compressedDataBuffer.begin() + compressedSize;
    compressedDataBuffer.erase(endDataIterator,compressedDataBuffer.end());

    // XXX: not sure if this holds for BTC
    //sometimes even if the tensor is > 4KB it might not be compressable
    if(compressedSize >= uncompressedDataSize)
    {
        toReturn.first = data;
        toReturn.second = uncompressedDataSize;
        return toReturn;
    }
    else
    {
        t.set<int>("CompressedSize", compressedSize);
        std::vector<int64_t> compressedData(compressedDataBuffer.begin(),compressedDataBuffer.end());
        toReturn.first = compressedData;
        toReturn.second = compressedSize;
        return toReturn;
    }
}

// XXX: Not sure if this API is needed
std::pair<std::vector<int64_t>, uint32_t> BTC::compress(std::vector<int64_t>& data, mv::Data::TensorIterator& /*t*/)
{
    if(data.empty()) {
        throw mv::ArgumentError("hde", "hdeCompress", "0", "Empty data vector");
    }

    BitCompactor::btcmpctr_compress_wrap_args_t btcArgs;

    btcArgs.bypass_en      = codec_->mBitCompactorConfig->bypass_en;
    btcArgs.dual_encode_en = codec_->mBitCompactorConfig->dual_encode_en;
    btcArgs.proc_bin_en    = codec_->mBitCompactorConfig->proc_bin_en;
    btcArgs.proc_btmap_en  = codec_->mBitCompactorConfig->proc_btmap_en;
    btcArgs.align          = codec_->mBitCompactorConfig->align;
    btcArgs.verbosity      = codec_->mBitCompactorConfig->verbosity;
    btcArgs.SblkSize       = codec_->mBitCompactorConfig->blockSize;
    btcArgs.LblkSize       = codec_->mBitCompactorConfig->superBlockSize;
    btcArgs.mixedBlkSize   = codec_->mBitCompactorConfig->mixedBlkSize;
    btcArgs.minFixedBitLn  = codec_->mBitCompactorConfig->minFixedBitLn;

    std::pair<std::vector<int64_t>, uint32_t> toReturn;
    std::vector<uint8_t> uncompressedData(data.begin(),data.end());
    uint32_t uncompressedDataSize = uncompressedData.size();

    auto compressedBufferSizeBound = codec_->btcmpctr_cmprs_bound(uncompressedDataSize);
    std::vector<uint8_t> compressedDataBuffer (compressedBufferSizeBound, 0);

    uint32_t compressedSize = codec_->CompressArray(&uncompressedData[0], uncompressedDataSize, &compressedDataBuffer[0], compressedBufferSizeBound, &btcArgs);

    std::vector<uint8_t>::iterator endDataIterator = compressedDataBuffer.begin() + compressedSize;
    compressedDataBuffer.erase(endDataIterator,compressedDataBuffer.end());

    //sometimes even if the tensor is > 4KB it might not be compressable
    if(compressedSize > uncompressedDataSize)
    {
        toReturn.first = data;
        toReturn.second = uncompressedDataSize;
        return toReturn;
    }
    else
    {
        std::vector<int64_t> compressedData(compressedDataBuffer.begin(),compressedDataBuffer.end());
        toReturn.first = compressedData;
        toReturn.second = compressedSize;
        return toReturn;
    }
}

std::vector<uint8_t> BTC::decompress(std::vector<uint8_t>& compressedData)
{
    BitCompactor::btcmpctr_compress_wrap_args_t btcArgs;

    btcArgs.bypass_en      = codec_->mBitCompactorConfig->bypass_en;
    btcArgs.dual_encode_en = codec_->mBitCompactorConfig->dual_encode_en;
    btcArgs.proc_bin_en    = codec_->mBitCompactorConfig->proc_bin_en;
    btcArgs.proc_btmap_en  = codec_->mBitCompactorConfig->proc_btmap_en;
    btcArgs.align          = codec_->mBitCompactorConfig->align;
    btcArgs.verbosity      = codec_->mBitCompactorConfig->verbosity;
    btcArgs.SblkSize       = codec_->mBitCompactorConfig->blockSize;
    btcArgs.LblkSize       = codec_->mBitCompactorConfig->superBlockSize;
    btcArgs.mixedBlkSize   = codec_->mBitCompactorConfig->mixedBlkSize;
    btcArgs.minFixedBitLn  = codec_->mBitCompactorConfig->minFixedBitLn;

    uint32_t compressedSize = compressedData.size();
    auto decompressedBufferSizeBound = compressedData.size() * BTC_MAX_DECOMPRESS_FACTOR;
    std::vector<uint8_t> decompressedDataBuffer (decompressedBufferSizeBound, 0);

    int decompressedSize = codec_->DecompressArray(&compressedData[0], compressedSize, &decompressedDataBuffer[0], decompressedBufferSizeBound, &btcArgs);

    std::vector<uint8_t>::iterator endDataIterator = decompressedDataBuffer.begin() + decompressedSize;
    decompressedDataBuffer.erase(endDataIterator, decompressedDataBuffer.end());

    return decompressedDataBuffer;
}
}
