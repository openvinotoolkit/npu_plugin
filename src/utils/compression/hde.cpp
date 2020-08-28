#include "include/mcm/utils/compression/hde.hpp"


namespace mv
{
Hde::Hde(uint32_t bitPerSymbol, uint32_t maxNumberEncodedSymbols, uint32_t verbosity, uint32_t blockSize, bool pStatsOnly, uint32_t bypassMode)
{
    codec_.reset(new huffmanCodec(bitPerSymbol, maxNumberEncodedSymbols, verbosity, blockSize, pStatsOnly, bypassMode));
}

std::pair<std::vector<int64_t>, uint32_t> Hde::hdeCompress(std::vector<int64_t>& data, mv::Tensor& t)
{
    std::pair<std::vector<int64_t>, uint32_t> toReturn;
    std::vector<uint8_t> uncompressedData(data.begin(),data.end());
    uint32_t uncompressedDataSize = uncompressedData.size();
    auto compressedBufferSize = uncompressedDataSize + 2 * (std::ceil(uncompressedDataSize / 4096.0) + 1);

    std::vector<uint8_t> compressedDataBuffer (compressedBufferSize, 0); 
    uint32_t compressedSize = codec_->huffmanCodecCompressArray(uncompressedDataSize, &uncompressedData[0], &compressedDataBuffer[0]);
    vector<uint8_t>::iterator endDataIterator = compressedDataBuffer.begin() + compressedSize;
    compressedDataBuffer.erase(endDataIterator,compressedDataBuffer.end());
    
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

std::pair<std::vector<int64_t>, uint32_t> Hde::hdeCompress(std::vector<int64_t>& data, mv::Data::TensorIterator& t)
{
    std::pair<std::vector<int64_t>, uint32_t> toReturn;
    std::vector<uint8_t> uncompressedData(data.begin(),data.end());
    uint32_t uncompressedDataSize = uncompressedData.size();
    auto compressedBufferSize = uncompressedDataSize + 2 * (std::ceil(uncompressedDataSize / 4096.0) + 1);

    std::vector<uint8_t> compressedDataBuffer (compressedBufferSize, 0); 
    uint32_t compressedSize = codec_->huffmanCodecCompressArray(uncompressedDataSize, &uncompressedData[0], &compressedDataBuffer[0]);
    vector<uint8_t>::iterator endDataIterator = compressedDataBuffer.begin() + compressedSize;
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

std::vector<uint8_t> Hde::hdeDecompress(std::vector<uint8_t>& compressedData)
{
    uint32_t comprssedSize = compressedData.size();
    auto deCompressedBufferSize = compressedData.size() * 5;
    std::vector<uint8_t> deCompressedDataBuffer (deCompressedBufferSize, 0); 
    codec_->huffmanCodecDecompressArray(comprssedSize, &compressedData[0], &deCompressedDataBuffer[0]);

    return deCompressedDataBuffer;
}
}