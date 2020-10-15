#include "mcm/tensor/shape.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/tensor/dtype/dtype.hpp"
#include "mcm/target/target_descriptor.hpp"
#include "include/huffman_encoding/Huffman.hpp"
#include "include/huffman_encoding/huffmanCodec.hpp"
#include <stdlib.h> 
#include "gtest/gtest.h"

using namespace testing;

TEST(hde, self_test)
{
    int randomSize;
    int randomNumber;
    std::vector<uint8_t> uncompressedData;
    std::unique_ptr<huffmanCodec> codec_(new huffmanCodec(8, 16, 0, 4096, false, false));

    /* initialize random seed: */
    srand (time(NULL));

    /* generate random number between 1000 and 10000 for array size: */
    randomSize = rand() % 10000 + 1000;

    for(unsigned i = 0; i < randomSize; ++i) 
    {
        randomNumber = rand() % 255 + 0; /* generate random uint8 number*/
        uncompressedData.push_back(randomNumber);
    }

    /*compress data*/
    auto compressedBufferSize = uncompressedData.size() + 2 * (std::ceil(uncompressedData.size() / 4096) + 1);
    uint32_t size = uncompressedData.size();
    std::vector<uint8_t> compressedData (compressedBufferSize, 0); 
    codec_->huffmanCodecCompressArray(size, &uncompressedData[0], &compressedData[0]);

    /*decompress data*/
    auto deCompressedBufferSize = compressedData.size() * 5;
    std::vector<uint8_t> deCompressedDataBuffer (deCompressedBufferSize, 0); 
    codec_->huffmanCodecDecompressArray(size, &compressedData[0], &deCompressedDataBuffer[0]);

    /*test that decompressed data equals original data*/
    for(unsigned i = 0; i < randomSize; ++i)
        ASSERT_EQ(uncompressedData[i], deCompressedDataBuffer[i]);
}
